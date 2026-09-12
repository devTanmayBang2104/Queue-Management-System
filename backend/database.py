"""
Database layer using SQLAlchemy + SQLite.
Stores tracking logs, queue statistics, and alert history.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Database file in /data directory
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "queue_analytics.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


class TrackingLog(Base):
    """Individual person tracking records."""
    __tablename__ = "tracking_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, index=True)
    frame_number = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    x = Column(Float)
    y = Column(Float)
    w = Column(Float)
    h = Column(Float)
    in_queue = Column(Boolean, default=False)
    queue_id = Column(String(50), nullable=True)
    velocity = Column(Float, default=0.0)


class QueueStat(Base):
    """Periodic queue statistics snapshots."""
    __tablename__ = "queue_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    queue_id = Column(String(50), default="default")
    queue_length = Column(Integer, default=0)
    avg_wait_time = Column(Float, default=0.0)
    max_wait_time = Column(Float, default=0.0)
    predicted_wait_time = Column(Float, default=0.0)
    arrival_rate = Column(Float, default=0.0)
    service_rate = Column(Float, default=0.0)
    abandonment_rate = Column(Float, default=0.0)


class AlertHistory(Base):
    """Historical alerts log."""
    __tablename__ = "alerts_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    alert_type = Column(String(50))
    message = Column(Text)
    severity = Column(String(20))
    threshold_value = Column(Float, nullable=True)
    actual_value = Column(Float, nullable=True)


def init_db():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Get a database session."""
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


# Initialize on import
init_db()
