import hashlib
import threading
from datetime import datetime
from os import environ
from typing import Any, Dict

import bson
from ming import create_datastore, schema
from ming.odm import FieldProperty, MappedClass, Mapper, ODMSession

DBHOST = environ[
    'DBHOST'] if 'DBHOST' in environ else 'mongodb://localhost:27017'
DATABASE_NAME = environ[
    'DATABASE_NAME'] if 'DATABASE_NAME' in environ else 'classifier_dev'

datastore = create_datastore(DBHOST + '/' + DATABASE_NAME)
session = ODMSession(bind=datastore, autoflush=False)
sessionLock = threading.Lock()


def hasher(seq: str) -> str:
    """MongoDb can't index sequences > 1024b, so we index hashes instead."""
    return hashlib.md5(str.encode(seq)).hexdigest()


def ValueToJson(value: Any) -> Any:
    if isinstance(value, bson.objectid.ObjectId):
        return str(value)
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, list):
        return [ValueToJson(member) for member in value]
    elif isinstance(value, dict):
        dict_data = {}
        for member in value:
            dict_data.update({member: ValueToJson(value[member])})
        return dict_data
    else:
        return value


class JsonOdmHelper:

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            k: ValueToJson(self.__getattribute__(k))
            for k in dir(type(self))
            if k not in dir(MappedClass) and k not in dir(JsonOdmHelper) and
            self.__getattribute__(k) is not None
        }


class Embedding(MappedClass, JsonOdmHelper):
    """Python representation of embedding cache schema in MongoDB."""

    class __mongometa__:
        session = session
        name = 'embedding_cache'
        unique_indexes = [('bert', 'seqHash')]

    _id = FieldProperty(schema.ObjectId)
    bert = FieldProperty(schema.String)
    seq = FieldProperty(schema.String)
    seqHash = FieldProperty(schema.String)
    embedding = FieldProperty(schema.Binary)
    update_timestamp = FieldProperty(datetime, if_missing=datetime.utcnow)


class ClassificationSample(MappedClass, JsonOdmHelper):
    """Python classification sample (training and/or predicted) in MongoDB."""

    class __mongometa__:
        session = session
        name = 'classification_sample'
        indexes = [('model',)]
        unique_indexes = [('model', 'seqHash')]

    _id = FieldProperty(schema.ObjectId)
    model = FieldProperty(schema.String)
    seq = FieldProperty(schema.String)
    seqHash = FieldProperty(schema.String)
    training_labels = FieldProperty(
        schema.Array(schema.Object(fields={'topic': schema.String})))
    # Keep separate to control which samples should
    # be used even if more have training_labels.
    use_for_training = FieldProperty(schema.Bool)
    predicted_labels = FieldProperty(
        schema.Array(
            schema.Object(fields={
                'topic': schema.String,
                'quality': schema.Float
            })))
    update_timestamp = FieldProperty(datetime, if_missing=datetime.utcnow)


Mapper.compile_all()
Mapper.ensure_all_indexes()
