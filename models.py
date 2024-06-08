from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()

class InformTable(db.Model):
    personid = db.Column(db.Integer, primary_key=True)
    gender = db.Column(db.String(10))
    LR = db.Column(db.String(10))

    def __init__(self, personid, gender, lr):
        self.personid = personid
        self.gender = gender
        self.lr = lr

