from datetime import datetime

from ming import schema
from ming.odm import FieldProperty, MappedClass, Mapper

from app.db import session


class Embedding(MappedClass):
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


class ClassificationSample(MappedClass):
    """Python representation of a classification sample (training and/or predicted) in MongoDB."""

    class __mongometa__:
        session = session
        name = 'classification_sample'
        indexes = [('model',)]
        unique_indexes = [('model', 'seqHash')]

    _id = FieldProperty(schema.ObjectId)
    model = FieldProperty(schema.String, required=True)
    seq = FieldProperty(schema.String, required=True)
    seqHash = FieldProperty(schema.String, required=True)
    training_labels = FieldProperty(schema.Array(schema.Object(fields={'topic': schema.String})))
    predicted_labels = FieldProperty(
        schema.Array(schema.Object(fields={
            'topic': schema.String,
            'probability': schema.Float
        })))
    update_timestamp = FieldProperty(datetime, if_missing=datetime.utcnow)


Mapper.compile_all()
Mapper.ensure_all_indexes()
