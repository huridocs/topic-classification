from os import environ
import hashlib
from ming import create_datastore
from ming.odm import ThreadLocalODMSession

DBHOST = environ['DBHOST'] if 'DBHOST' in environ else 'mongodb://localhost:27017'
DATABASE_NAME = 'classifier_dev'

session = ThreadLocalODMSession(bind=create_datastore(DBHOST + '/' + DATABASE_NAME),
                                autoflush=False)


def hasher(seq: str) -> str:
    return hashlib.md5(str.encode(seq)).hexdigest()
