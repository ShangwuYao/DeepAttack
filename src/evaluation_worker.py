import boto3
import json
import time
import os, sys
import warnings
#print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import db, Submission
from evaluation import Model_Evaluator


try:
    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')
    bucket = s3.Bucket('advex')

    sqs = boto3.client('sqs')
    resp = sqs.get_queue_url(QueueName='advex')
    queue_url = resp['QueueUrl']
except:
    # should raise error here
    warnings.warn("sqs not started", UserWarning)

# SAMPLE_FEEDBACK = {
#   "robustness": "9",
#   "rating": "Good",
#   "details": {
#       "original_accuracy": "98.55%",
#       "attack_results": [
#           {
#               "attack_method": "FGSM",
#               "accuracy": "80.05%",
#               "confidence": "95%"
#           },
#           {
#               "attack_method": "Basic Iterative Method",
#               "accuracy": "92.10%",
#               "confidence": "91%"
#           },
#           {
#               "attack_method": "Carlini Wagner",
#               "accuracy": "94.10%",
#               "confidence": "93%"
#           },
#           {
#               "attack_method": "Momentum Iterative Method",
#               "accuracy": "94.10%",
#               "confidence": "93.7%"
#           },
#           {
#               "attack_method": "DeepFool",
#               "accuracy": "90.10%",
#               "confidence": "89%"
#           }
#       ]
#   },
#   "suggestion": "Your model can be made more robust by training it with some of the adversarial examples which you can download for free from your dashboard."
# }


def update_feedback(submission_id, feedback=None, status=None):
    print('Writing feedback.')
    submission = Submission.query.get(submission_id)
    if feedback is not None:
        submission.feedback = feedback
    if status is not None:
        submission.status = status
    db.session.commit()


def evaluate_job(job):
    print('Evaluating model.')

    feedback={}
    submission_id = job['submission_id']
    model_file = job['s3_model_key']
    index_file = job['s3_index_key']

    update_feedback(submission_id, status='Running')
    
    #Check 1: File extension
    if not model_file.endswith('.h5'):
        if not index_file.endswith('.json'):
            feedback = {"error": "Model file has to have .h5 as its extension. Index file has to have .json as its extension"}
        else:
            feedback = {"error": "Model file has to have .h5 as its extension."}
    else:
        if not index_file.endswith('.json'):
            feedback = {"error": "Index file has to have .json as its extension"}       
    
    response_model = s3_client.head_object(Bucket='advex', Key=model_file)
    response_index = s3_client.head_object(Bucket='advex', Key=index_file)

    model_size=response_model['ContentLength']
    index_size=response_index['ContentLength']

    bucket.download_file(model_file, model_file)
    bucket.download_file(index_file, index_file)

    #Check 2: File Size Check
    if not feedback:
        if model_size > 1073741824: # 1 GiB
            if index_size > 102400:
                feedback = {"error": ".h5 file can't be bigger than 1GB and .json file can't be bigger than 100KB."}
            else:
                feedback = {"error": ".h5 file can't be bigger than 1GB."}
        else:
            if index_size > 102400:
                feedback = {"error": ".json file can't be bigger than 100KB."}
    
    if not feedback:
        #The model file and index file are perfectly fine.
        try:
            model=Model_Evaluator(model_file,index_file)
            feedback=model.evaluate()
        except Exception as exc:
            feedback['error']=exc.__str__()
    
    print(feedback)
    status = ('error' in feedback ? 'Failed' : 'Finished')
    write_feedback(submission_id, feedback=feedback, status=status)
    
def main():
    while True:
        resp = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1)
        if 'Messages' not in resp:
            print('No messages received, sleep for 10s.')
            time.sleep(10)
            continue

        print('Message received.')
        message = resp['Messages'][0]
        receipt_handle = message['ReceiptHandle']
        job = json.loads(message['Body'])

        # Process job
        evaluate_job(job)

        # Delete message
        resp = sqs.delete_message(QueueUrl=queue_url,ReceiptHandle=receipt_handle)


if __name__ == '__main__':
    main()
