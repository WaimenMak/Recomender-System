import sqlite3
from sqlite3 import Error
class SQLiteConnection: 
    _instance = None 

    def __new__(cls):
        
        #Singelton Pattern -> Prevents multiple connections from opening 
         if cls._instance is None:
            #Creates new instance of SQLConnection 
            cls._instance = object.__new__(cls)
            #Getting connection data from KeyVault 
            #Tries to create connection and cursor objects -> are stored in the instance 
            try:
              
                newConnection = sqlite3.connect('recDatabase.db',check_same_thread=False)
                newCursor = newConnection.cursor()
                SQLiteConnection._instance.cnxn = newConnection
                SQLiteConnection._instance.crsr = newCursor
            except Error as e:
                print(e)

         else:
            print('connection established\n{}')

         return cls._instance

    #Every new initiated SQLConnection uses the same connection and cursor -> 
    def __init__(self):
        self._cnxn = self._instance.cnxn
        self._cursor = self._instance.crsr

    @property
    def connection(self):
        return self._cnxn

    @property
    def cursor(self):
        return self._cursor

    def insert_statement(self, sql_statement): 
        try: 
            self._cursor.execute(sql_statement) 
            self._cnxn.commit()

        except Error as  e: 
            print("Exception during statement exection: ", e)
        

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()




 