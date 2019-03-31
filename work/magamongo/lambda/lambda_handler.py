import json
import numpy as np
import boto3
from PIL import Image
from io import BytesIO
import base64

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            if abs(o) % 1 > 0:
                return float(o)
            else:
                return int(o)
        return super(DecimalEncoder, self).default(o)

def lambda_handler(event, context):
    # request body
    posted = json.loads(event["body"])
    print("bodyのなかみ: ", posted)
    # image = base64.b64decode(posted["pic"])
    image = base64.b64decode( posted["pic"].split(',')[1] )
    image = Image.open(BytesIO(image))
    #resize
    image = np.array(image.resize((320,320)), dtype='float').reshape(1,320,320,3)
    print("処理前の画像", image)
    image = image-128
    print("処理中の画像", image)
    image = (image/128).tolist()
    print("sagemakerに投げる画像", image)
    # dummypic = np.round(np.random.rand(1,320,320,3), decimals=1).tolist()
    domain = posted["domain"]
    print("フロントから来たドメイン", domain)
    # dummydomain = np.random.randint(-1,2,(1,8)).tolist()
    
    req = {
        "input_image": image,
        "input_domain": domain
      }
  
    req = json.dumps(req)
    runtime = boto3.client('sagemaker-runtime')
    EPname = "sagemaker-tensorflow-2019-03-30-01-54-07-845"
    print(domain)
    print("endopoint calling..")
    response = runtime.invoke_endpoint(EndpointName=EPname, ContentType="application/json", Accept="application/json", Body=req)
    response = json.load(response["Body"])['outputs']['output_image']['floatVal']
    # response = ['{:.2f}'.format(n) for n in response]
    res = np.array([(item*128+128) for item in response], dtype='uint8').reshape(320,320,3)
    print(res)
    res = Image.fromarray(res)
    buff = BytesIO()
    res.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode('utf-8')
    # res = base64.b64encode(res).decode('utf-8')
    print("result: ", img_str)
    return {
        'statusCode': '200',
        'body': img_str,
        'headers': {
            'Content-Type': 'application/jpeg',
            'Access-Control-Allow-Origin': '*'
        },
        'isBase64Encoded': True
    }