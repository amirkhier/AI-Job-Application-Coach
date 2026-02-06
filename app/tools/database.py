import mysql.connector
from mysql.connector import Error
import json
import os
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database connection and operations manager for MySQL."""
    
    def __init__(self):
        """Initialize database connection parameters from environment variables."""
        self.host = os.getenv('MYSQL_HOST', 'localhost')
        self.port = int(os.getenv('MYSQL_PORT', 3306))
        self.user = os.getenv('MYSQL_USER', 'root')
        self.password = os.getenv('MYSQL_PASSWORD', '')
        self.database = os.getenv('MYSQL_DATABASE', 'job_coach')
        self.connection = None
        
    def connect(self):
        """Establish database connection."""
        try:
            if self.connection is None or not self.connection.is_connected():
                self.connection = mysql.connector.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    charset='utf8mb4',
                    use_unicode=True,
                    autocommit=False
                )
                logger.info("Successfully connected to MySQL database")
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            self.connection = None
            logger.info("MySQL connection closed")
    
    def ensure_connection(self):
        """Ensure database connection is active."""
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
        except Error:
            self.connect()
    
    def execute_query(self, query: str, params: tuple = None) -> Optional[List[Dict]]:
        """Execute a SELECT query and return results as list of dictionaries."""
        try:
            self.ensure_connection()
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            logger.error(f"Error executing query: {e}")
            return None
    
    def execute_update(self, query: str, params: tuple = None) -> Optional[int]:
        """Execute INSERT/UPDATE/DELETE query and return affected rows or last insert ID."""
        try:
            self.ensure_connection()
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            self.connection.commit()
            
            # For INSERT operations, return the last insert ID
            if query.strip().upper().startswith('INSERT'):
                result = cursor.lastrowid
            else:
                result = cursor.rowcount
                
            cursor.close()
            return result
        except Error as e:
            logger.error(f"Error executing update: {e}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def execute_many(self, query: str, params_list: List[tuple]) -> Optional[int]:
        """Execute multiple queries with different parameters."""
        try:
            self.ensure_connection()
            cursor = self.connection.cursor()
            cursor.executemany(query, params_list)
            self.connection.commit()
            affected_rows = cursor.rowcount
            cursor.close()
            return affected_rows
        except Error as e:
            logger.error(f"Error executing batch query: {e}")
            if self.connection:
                self.connection.rollback()
            return None
    
    # User management methods
    def create_user(self, email: str, profile_data: Dict = None, preferences: Dict = None) -> Optional[int]:
        """Create a new user and return the user ID."""
        query = """
        INSERT INTO users (email, profile_data, preferences) 
        VALUES (%s, %s, %s)
        """
        params = (
            email,
            json.dumps(profile_data) if profile_data else None,
            json.dumps(preferences) if preferences else None
        )
        return self.execute_update(query, params)
    
    def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user by ID."""
        query = "SELECT * FROM users WHERE id = %s"
        results = self.execute_query(query, (user_id,))
        if results:
            user = results[0]
            # Parse JSON fields
            if user.get('profile_data'):
                user['profile_data'] = json.loads(user['profile_data'])
            if user.get('preferences'):
                user['preferences'] = json.loads(user['preferences'])
            return user
        return None
    
    def update_user_profile(self, user_id: int, profile_data: Dict) -> bool:
        """Update user profile data."""
        query = "UPDATE users SET profile_data = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
        result = self.execute_update(query, (json.dumps(profile_data), user_id))
        return result is not None and result > 0
    
    # Conversation management methods
    def save_conversation(self, user_id: int, session_id: str, message: str, 
                         response: str = None, intent: str = None, 
                         agent_used: str = None, metadata: Dict = None) -> Optional[int]:
        """Save a conversation turn."""
        query = """
        INSERT INTO conversations (user_id, session_id, message, response, intent, agent_used, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            user_id, session_id, message, response, intent, agent_used,
            json.dumps(metadata) if metadata else None
        )
        return self.execute_update(query, params)
    
    def get_conversation_history(self, user_id: int, session_id: str = None, limit: int = 20) -> List[Dict]:
        """Get conversation history for a user, optionally filtered by session."""
        if session_id:
            query = """
            SELECT * FROM conversations 
            WHERE user_id = %s AND session_id = %s 
            ORDER BY created_at DESC LIMIT %s
            """
            params = (user_id, session_id, limit)
        else:
            query = """
            SELECT * FROM conversations 
            WHERE user_id = %s 
            ORDER BY created_at DESC LIMIT %s
            """
            params = (user_id, limit)
        
        results = self.execute_query(query, params)
        if results:
            # Parse JSON metadata
            for conv in results:
                if conv.get('metadata'):
                    conv['metadata'] = json.loads(conv['metadata'])
        return results or []
    
    # Interview session management
    def create_interview_session(self, user_id: int, session_id: str, role: str, level: str) -> Optional[int]:
        """Create a new interview session."""
        query = """
        INSERT INTO interview_sessions (user_id, session_id, role, level, questions, answers, feedback)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        params = (user_id, session_id, role, level, json.dumps([]), json.dumps([]), json.dumps({}))
        return self.execute_update(query, params)
    
    def update_interview_session(self, session_id: str, questions: List = None, 
                                answers: List = None, feedback: Dict = None, 
                                score: float = None, completed: bool = False) -> bool:
        """Update interview session data."""
        updates = []
        params = []
        
        if questions is not None:
            updates.append("questions = %s")
            params.append(json.dumps(questions))
        if answers is not None:
            updates.append("answers = %s")
            params.append(json.dumps(answers))
        if feedback is not None:
            updates.append("feedback = %s")
            params.append(json.dumps(feedback))
        if score is not None:
            updates.append("score = %s")
            params.append(score)
        if completed:
            updates.append("completed_at = CURRENT_TIMESTAMP")
        
        if not updates:
            return False
        
        query = f"UPDATE interview_sessions SET {', '.join(updates)} WHERE session_id = %s"
        params.append(session_id)
        
        result = self.execute_update(query, tuple(params))
        return result is not None and result > 0
    
    def get_interview_session(self, session_id: str) -> Optional[Dict]:
        """Get interview session by session ID."""
        query = "SELECT * FROM interview_sessions WHERE session_id = %s"
        results = self.execute_query(query, (session_id,))
        if results:
            session = results[0]
            # Parse JSON fields
            for field in ['questions', 'answers', 'feedback']:
                if session.get(field):
                    session[field] = json.loads(session[field])
            return session
        return None
    
    # Application tracking methods
    def create_application(self, user_id: int, company_name: str, position_title: str,
                          job_url: str = None, status: str = 'applied',
                          application_date: datetime = None, notes: str = None) -> Optional[int]:
        """Create a new job application record."""
        query = """
        INSERT INTO applications (user_id, company_name, position_title, job_url, status, application_date, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        if application_date is None:
            application_date = datetime.now().date()
            
        params = (user_id, company_name, position_title, job_url, status, application_date, notes)
        return self.execute_update(query, params)
    
    def get_applications(self, user_id: int, status: str = None) -> List[Dict]:
        """Get applications for a user, optionally filtered by status."""
        if status:
            query = """
            SELECT * FROM applications 
            WHERE user_id = %s AND status = %s 
            ORDER BY application_date DESC
            """
            params = (user_id, status)
        else:
            query = """
            SELECT * FROM applications 
            WHERE user_id = %s 
            ORDER BY application_date DESC
            """
            params = (user_id,)
        
        return self.execute_query(query, params) or []
    
    def update_application_status(self, application_id: int, status: str, notes: str = None) -> bool:
        """Update application status and optionally add notes."""
        if notes:
            query = "UPDATE applications SET status = %s, notes = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
            params = (status, notes, application_id)
        else:
            query = "UPDATE applications SET status = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
            params = (status, application_id)
        
        result = self.execute_update(query, params)
        return result is not None and result > 0


# Global database instance
db = DatabaseManager()


def get_db() -> DatabaseManager:
    """Get the global database instance."""
    return db


def init_db():
    """Initialize database connection. Call this on app startup."""
    try:
        db.connect()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def close_db():
    """Close database connection. Call this on app shutdown."""
    try:
        db.disconnect()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")


# Context manager for database operations
class DatabaseTransaction:
    """Context manager for database transactions."""
    
    def __init__(self, db_instance: DatabaseManager):
        self.db = db_instance
        
    def __enter__(self):
        self.db.ensure_connection()
        return self.db
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if self.db.connection:
                self.db.connection.rollback()
        # Don't close connection, let the manager handle it