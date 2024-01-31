# database.py

import mysql.connector
from mysql.connector import Error

class Database:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            return self.connection.is_connected()
        except Error as e:
            print(f"Error connecting to MySQL database: {e}")
            return False

    def execute_query(self, query, params=None):
        if self.connection and self.connection.is_connected():
            cursor = self.connection.cursor(buffered=True)
            try:
                cursor.execute(query, params or ())
                self.connection.commit()
                return cursor.fetchall()  # Fetch all results
            except Error as e:
                print(f"Error executing query: {e}")
                return None
            finally:
                cursor.close()
        else:
            print("Failed to execute query: No connection to the database.")
            return None


    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

