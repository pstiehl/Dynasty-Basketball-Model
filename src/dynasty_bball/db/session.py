"""DB engine, session factory, and init helper."""
from __future__ import annotations
from contextlib import contextmanager
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, Session
from .models import Base
from ..config import settings

engine = create_engine(settings.database_url, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create all tables. Idempotent. Also runs lightweight migrations for
    columns added in later versions, so old DBs upgrade cleanly."""
    Base.metadata.create_all(bind=engine)
    _migrate()


def _migrate() -> None:
    """Add new columns to existing tables. SQLite only — for Postgres use Alembic."""
    if not settings.database_url.startswith("sqlite"):
        return

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    if "players" in tables:
        player_cols = {c["name"] for c in inspector.get_columns("players")}
        with engine.begin() as conn:
            if "normalized_name" not in player_cols:
                conn.execute(text("ALTER TABLE players ADD COLUMN normalized_name VARCHAR(128)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_players_normalized_name ON players(normalized_name)"))
            if "est_retirement_age" not in player_cols:
                conn.execute(text("ALTER TABLE players ADD COLUMN est_retirement_age FLOAT"))
            if "years_remaining" not in player_cols:
                conn.execute(text("ALTER TABLE players ADD COLUMN years_remaining FLOAT"))

    if "composite_scores" in tables:
        cols = {c["name"] for c in inspector.get_columns("composite_scores")}
        with engine.begin() as conn:
            if "consensus_rank" not in cols:
                conn.execute(text("ALTER TABLE composite_scores ADD COLUMN consensus_rank INTEGER"))
            if "rank_divergence" not in cols:
                conn.execute(text("ALTER TABLE composite_scores ADD COLUMN rank_divergence INTEGER"))


@contextmanager
def get_session() -> Session:
    """Context-managed session with commit/rollback."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
