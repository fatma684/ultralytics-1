"""Database Models - SQLAlchemy models for PostgreSQL."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Event(Base):
    """Event table for storing detection events."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    camera_id = Column(String(50), index=True)
    track_id = Column(Integer)
    event_type = Column(String(50), index=True)  # entry, exit, region_enter, etc.
    class_name = Column(String(100))
    confidence = Column(Float)
    x = Column(Float)
    y = Column(Float)
    x_min = Column(Float)
    y_min = Column(Float)
    x_max = Column(Float)
    y_max = Column(Float)
    region_name = Column(String(100), nullable=True)
    metadata = Column(Text, nullable=True)  # JSON string

    __table_args__ = (
        Index("ix_events_camera_timestamp", "camera_id", "timestamp"),
        Index("ix_events_type_timestamp", "event_type", "timestamp"),
    )


class CameraSession(Base):
    """Camera tracking session."""

    __tablename__ = "camera_sessions"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(50), unique=True, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_entries = Column(Integer, default=0)
    total_exits = Column(Integer, default=0)
    peak_crowd = Column(Integer, default=0)
    status = Column(String(20), default="active")  # active, inactive


class RegionStats(Base):
    """Statistics for specific regions."""

    __tablename__ = "region_stats"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    camera_id = Column(String(50), index=True)
    region_name = Column(String(100), index=True)
    entry_count = Column(Integer, default=0)
    exit_count = Column(Integer, default=0)
    current_count = Column(Integer, default=0)

    __table_args__ = (Index("ix_region_stats_camera_region", "camera_id", "region_name"),)


# Database configuration
DATABASE_URL = "postgresql://user:password@localhost/event_tracking"


def get_db_url(user: str = "postgres", password: str = "postgres", host: str = "localhost", port: int = 5432, db: str = "event_tracking") -> str:
    """Generate database URL."""
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def init_db(db_url: str = DATABASE_URL) -> None:
    """Initialize database tables."""
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized: {db_url}")


def get_session(db_url: str = DATABASE_URL):
    """Get database session."""
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()
