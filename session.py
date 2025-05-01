from flask import session
def setSession(username):
    session['username'] = username
    session.permanent = True

def getSession():
    username = session.get('username')
    return username

def delSession():
    session.pop('username')