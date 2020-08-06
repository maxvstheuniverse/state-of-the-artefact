import boto3
import compress_pickle as cp
from botocore.exceptions import ClientError


def upload_obj(obj, bucket, object_name, compression="gzip"):
    """ Upload a Python Object to an S3 bucket. """
    buffer = cp.dumps(obj, compression=compression)
    return upload_buffer(buffer, bucket, object_name)


def upload_buffer(buffer, bucket, object_name):
    """ Upload a buffer to an S3 bucket. """
    s3 = boto3.resource('s3')
    file_obj = s3.Object(bucket, object_name)

    try:
        file_obj.put(Body=buffer)
    except ClientError as e:
        return 1
    return 0
