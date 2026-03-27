import boto3
import os

def upload_images(image_paths, prefix):
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    bucket = os.getenv("S3_BUCKET_NAME")

    urls = []
    for image in image_paths:
        key = f"{prefix}/{os.path.basename(image)}"
        s3.upload_file(image, bucket, key, ExtraArgs={"ContentType": "image/png"})
        urls.append(f"https://{bucket}.s3.amazonaws.com/{key}")

    return urls
