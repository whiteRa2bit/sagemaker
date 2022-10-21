import sagemaker
import botocore
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets.filesystems import S3FileSystem
from sagemaker.huggingface import HuggingFace


def create_session():
    sess = sagemaker.Session()
    sagemaker_session_bucket = None
    if sagemaker_session_bucket is None and sess is not None:
        sagemaker_session_bucket = sess.default_bucket()

    role = 'SageMakerRole'
    sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)
    return sess, role



def get_datset():
    # load dataset
    dataset = load_dataset("xed_en_fi", "en_annotated")
    dataset = dataset['train']

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # create tokenization function
    def tokenize(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True)

    # prepare labels
    def convert_labels(batch, num_labels=8):
        binary_labels = []
        for labels in batch['labels']:
            cur_binary_labels = np.bincount(labels).astype(float) 
            cur_binary_labels.resize(num_labels)
            binary_labels.append(cur_binary_labels)
        return {'labels': binary_labels}

    # Preprocess dataset
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.map(convert_labels, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Train-test split
    train_test_dataset = dataset.train_test_split(test_size=0.02)
    train_dataset = train_test_dataset['train']
    test_dataset = train_test_dataset['test']

    return train_dataset, test_dataset


def upload_dataset(sess, train_dataset, test_dataset):
    s3_prefix = 'samples/datasets/xed_en_fi'
    s3 = S3FileSystem()

    # save train_dataset to S3
    training_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/train'
    # train_dataset.save_to_disk(training_input_path,fs=s3)

    # save test_dataset to S3
    test_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/test'
    # test_dataset.save_to_disk(test_input_path,fs=s3)

    return training_input_path, test_input_path


def train(training_input_path, test_input_path, role):
    hyperparameters={
        "epochs": 1,                            # number of training epochs
        "train_batch_size": 32,                 # training batch size
        "model_name":"distilbert-base-uncased",  # name of pretrained model
    }

    huggingface_estimator = HuggingFace(
        entry_point="train.py",                 # fine-tuning script to use in training job
        source_dir="./scripts",                 # directory where fine-tuning script is stored
        instance_type="ml.p3.2xlarge",          # instance type
        instance_count=1,                       # number of instances
        role=role,                              # IAM role used in training job to acccess AWS resources (S3)
        transformers_version="4.6",             # Transformers version
        pytorch_version="1.7",                  # PyTorch version
        py_version="py36",                      # Python version
        hyperparameters=hyperparameters         # hyperparameters to use in training job
    )

    huggingface_estimator.fit({"train": training_input_path, "test": test_input_path})


def main():
    sess, role = create_session()
    train_dataset, test_dataset = get_datset()
    training_data_path, test_data_path = upload_dataset(sess, train_dataset, test_dataset)
    train(training_data_path, test_data_path, role)


if __name__ == '__main__':
    main()
