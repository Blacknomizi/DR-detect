import mysql.connector

def get_connection():
    cnx = mysql.connector.connect(user='root', password='123456', host='127.0.0.1', database='web')
    return cnx

def close_connection(cursor, cnx):
    cursor.close()
    cnx.close()

