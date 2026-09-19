import re
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

STOPWORDS = set('ich du er sie es wir ihr und oder aber der die das ein eine einem einer den dem des ist sind war wie was wer warum wann wo mich mir dich dir mein meine dein deine über fuer für mit von zu auf an im am in als hat haben habe hast sich nicht noch schon bitte weißt weisst kannst erinnern erinnerst what who why when where how i you he she it we they the a an and or but is are was were my your me about do does did know remember can could have has of to in on for with this that yesterday gestern heute today'.split())
FACT_PATTERNS = [
    ('name', 'fact', 0.95, r'^(?:ich heiße|ich heisse|mein name ist|my name is)\s+\S'),
    ('home', 'fact', 0.85, r'^(?:ich wohne|ich lebe|i live)\s+\S'),
    ('occupation', 'fact', 0.85, r'^(?:ich arbeite als|ich studiere|i work as|i study)\s+\S'),
    (None, 'preference', 0.75, r'^(?:ich bevorzuge|ich mag|ich möchte künftig|ich moechte kuenftig|i prefer|i like|i dislike)\s+\S'),
    (None, 'decision', 0.9, r'^(?:ich habe entschieden|wir haben entschieden|ich entscheide mich|i decided|we decided|we have decided)\s+\S'),
    (None, 'event', 0.8, r'^(?:mein geburtstag ist|my birthday is|mein termin ist|my appointment is)\s+\S'),
    (None, 'topic', 0.65, r'^(?:mein projekt ist|my project is|wir sprechen über|wir sprechen ueber|we are discussing|das thema ist)\s+\S'),
]


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def terms(text):
    return list(dict.fromkeys(t for t in re.findall(r'[^\W_]+', text.casefold()) if len(t) > 2 and t not in STOPWORDS))


def important_facts(text):
    for sentence in re.split(r'(?<=[.!?])\s+|\n+', text.strip()):
        if not 4 <= len(sentence) <= 600 or sentence.endswith('?'):
            continue
        for key, category, importance, pattern in FACT_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                yield sentence, key, category, importance
                break


@dataclass
class MemoryItem:
    id: int
    content: str
    created_at: str
    updated_at: str
    importance: float
    category: str
    source: str
    session_id: str
    fact_key: str | None
    active: int
    relevance: float = 0.0


class MemoryStore:
    def __init__(self, path='artifacts/memory.sqlite', session_id=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid.uuid4().hex
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY, content TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, importance REAL NOT NULL, category TEXT NOT NULL,
                source TEXT NOT NULL, session_id TEXT NOT NULL, fact_key TEXT, active INTEGER NOT NULL DEFAULT 1);
            CREATE INDEX IF NOT EXISTS memories_key ON memories(fact_key, active);
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, content=memories, content_rowid=id, tokenize='unicode61 remove_diacritics 2');
            CREATE TRIGGER IF NOT EXISTS memories_insert AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_delete AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
            END;
        ''')

    def close(self):
        self.db.close()

    def add(self, content, importance=0.8, category='fact', source='user', fact_key=None, created_at=None):
        content = content.strip()
        if not content or len(content) > 4000 or not 0 <= importance <= 1:
            raise ValueError('Invalid memory content/importance')
        stamp = datetime.fromisoformat(created_at or now_utc())
        if stamp.tzinfo is None:
            raise ValueError('Memory timestamp must include its timezone')
        stamp = stamp.astimezone(timezone.utc).isoformat(timespec='seconds')
        with self.db:
            row = self.db.execute('SELECT id FROM memories WHERE content=? AND active=1', (content,)).fetchone()
            if row:
                self.db.execute('UPDATE memories SET updated_at=? WHERE id=?', (stamp, row['id']))
                return row['id']
            if fact_key:
                self.db.execute('UPDATE memories SET active=0 WHERE fact_key=? AND active=1', (fact_key,))
            cursor = self.db.execute('INSERT INTO memories(content,created_at,updated_at,importance,category,source,session_id,fact_key) VALUES (?,?,?,?,?,?,?,?)',
                (content, stamp, stamp, importance, category, source, self.session_id, fact_key))
            return cursor.lastrowid

    def consider(self, text):
        return [self.add(content, importance, category, fact_key=key)
                for content, key, category, importance in important_facts(text)]

    def forget(self, memory_id):
        with self.db:
            return self.db.execute('DELETE FROM memories WHERE id=?', (memory_id,)).rowcount > 0

    def list(self, limit=50, include_history=False):
        rows = self.db.execute('SELECT * FROM memories '+('' if include_history else 'WHERE active=1 ')+
                               'ORDER BY updated_at DESC,id DESC LIMIT ?', (limit,)).fetchall()
        return [MemoryItem(**dict(row)) for row in rows]

    def search(self, query, limit=6):
        if limit < 1:
            return []
        words = terms(query)
        profile = bool(re.search(r'(?:über mich|ueber mich|about me|mein name|my name|wo wohne|where do i live|wer bin ich|who am i)', query, re.I))
        changed = bool(re.search(r'(?:verändert|veraendert|geändert|geaendert|changed|früher|frueher|before)', query, re.I))
        yesterday = bool(re.search(r'\b(?:gestern|yesterday)\b', query, re.I))
        active = '' if changed else ' AND m.active=1'
        rows = {}
        if words:
            match = ' OR '.join('"'+word+'"*' for word in words[:24])
            results = self.db.execute('SELECT m.*, bm25(memories_fts) AS rank FROM memories_fts '
                'JOIN memories m ON m.id=memories_fts.rowid WHERE memories_fts MATCH ?'+active+
                ' ORDER BY rank LIMIT 100', (match,)).fetchall()
            rows.update((r['id'], (r, 1/(1+idx))) for idx, r in enumerate(results))
        if profile:
            results = self.db.execute('SELECT m.* FROM memories m WHERE m.category IN (\'fact\',\'preference\',\'explicit\')'+active+
                ' ORDER BY m.importance DESC,m.updated_at DESC LIMIT 50').fetchall()
            rows.update((r['id'], (r, 0.8)) for r in results)
        if yesterday:
            day = datetime.now().astimezone().date()-timedelta(days=1)
            start = datetime.combine(day, datetime.min.time()).astimezone(timezone.utc).isoformat(timespec='seconds')
            end = datetime.combine(day+timedelta(days=1), datetime.min.time()).astimezone(timezone.utc).isoformat(timespec='seconds')
            results = self.db.execute('SELECT m.* FROM memories m WHERE m.created_at>=? AND m.created_at<?'+active+
                ' ORDER BY m.importance DESC LIMIT 100', (start, end)).fetchall()
            rows = {r['id']: (r, 1.0) for r in results}
        ranked = []
        now = datetime.now(timezone.utc)
        for row, lexical in rows.values():
            values = {key: row[key] for key in row.keys() if key != 'rank'}
            item = MemoryItem(**values)
            age = max(0, (now-datetime.fromisoformat(item.updated_at)).total_seconds()/86400)
            item.relevance = round(0.75*lexical+0.2*item.importance+0.05/(1+age/30), 5)
            ranked.append(item)
        return sorted(ranked, key=lambda item: (item.relevance, item.id), reverse=True)[:limit]

    def save_summary(self, messages):
        user_messages = [m['content'] for m in messages if m['role'] == 'user']
        if not user_messages:
            return None
        excerpts = []
        for text in user_messages[-8:]:
            excerpt = re.split(r'(?<=[.!?])\s+', text)[0][:240]
            excerpts.append('Nutzer: '+excerpt)
        return self.add('\n'.join(excerpts), 0.6, 'summary', 'user_excerpts')
