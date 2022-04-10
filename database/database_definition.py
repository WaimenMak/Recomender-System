import sqlite3
from sqlite3 import Error
import pandas as pd 


conn = sqlite3.connect('recDatabase.db')


# Table definitions----------------------------------------------------

# movie_info.csv 
movie_info = """  CREATE TABLE IF NOT EXISTS movie_info (
                                        movie_id integer PRIMARY KEY,
                                        movie_title text, 
                                        release_data text,
                                        IMdb text, 
                                        URL text, 
                                        unknown text, 
                                        Action text, 
                                        Adventure text, 
                                        Animation text, 
                                        Children text, 
                                        Comedy text, 
                                        Crime text, 
                                        Documentary text, 
                                        Drama text, 
                                        Fantasy text, 
                                        Film_Noir text, 
                                        Horror text, 
                                        Musical text, 
                                        Mystery text, 
                                        Romance text, 
                                        Sci_Fi text, 
                                        Thriller text, 
                                        War text, 
                                        Western text, 
                                        poster_url text)

"""
# movie-poster.csv 
movie_poster = """  CREATE TABLE IF NOT EXISTS movie_poster (
                                        movie_id integer PRIMARY KEY,
                                        poster_url text)
"""
# runtimeu_data_.csv 
runtime_u_data = """  CREATE TABLE IF NOT EXISTS runtime_u_data (
                                        user_id integer,
                                        movie_id integer NOT NULL, 
                                        score integer, 
                                        round integer, 
                                        algorithm integer, 
                                        FOREIGN KEY(movie_id) REFERENCES projects (id))

"""

## u.genre 
u_data = """  CREATE TABLE IF NOT EXISTS u_data (
                                        user_id integer,
                                        movie_id integer NOT NULL, 
                                        score integer, 
                                        timestamp integer,
                                        FOREIGN KEY(movie_id) REFERENCES projects (id)
"""
## To store the user profile of each user 
user_profile = """  CREATE TABLE IF NOT EXISTS user_profile (
                                        user_id integer PRIMARY KEY,
                                        Action real, 
                                        Adventure real, 
                                        Animation real, 
                                        Children real, 
                                        Comedy real, 
                                        Crime real, 
                                        Documentary real, 
                                        Drama real, 
                                        Fantasy real, 
                                        Film-Noir real, 
                                        Horror real, 
                                        Musical real, 
                                        Mystery text, 
                                        Romance real, 
                                        Sci-Fi real, 
                                        IMAX real, 
                                        Thriller real, 
                                        War real, 
                                        Western real )

"""
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def main():
    database = "recDatabase.db"


    # create a database connection
    conn = create_connection(database)

    # create tables
    if conn is not None:
        create_table(conn, movie_info)
        create_table(conn, movie_poster)
        create_table(conn, runtime_u_data)
        create_table(conn, u_data)
        create_table(conn, user_profile)
    else:
        print("Error! cannot create the database connection.")

    # Copy the data to the database
    movie_info_df = pd.read_csv("./data/movie_info_new.csv")
    # movie_poster_df = pd.read_csv("movie_poster.csv")
    user_data_df = pd.read_csv('./oldData//u.data')

    movie_info_df.to_sql(name='movie_info', con=conn, if_exists='replace')
    # movie_poster_df.to_sql(name='movie_poster', con=conn,if_exists='replace')
    user_data_df.to_sql(name='user_data', con=conn, if_exists='replace')

if __name__ == '__main__':
    main()
conn.close()