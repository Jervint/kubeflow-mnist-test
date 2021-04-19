# -*- encoding: utf-8 -*-
'''
Date: 2021-04-15 16:59:11
LastEditors: Jervint
LastEditTime: 2021-04-16 10:25:04
Description: 
FilePath: /kubeflow-pytorch-test/deployment/deploy_pipeline.py
'''
import kfp
from kfp import dsl
from kfp.dsl import PipelineVolume

def serving_op(image:str,bucket_name:str,model_name:str,model_version:str):
    namespace='kfserving-inference-service'
    runtime_version='2.0.0'
    service_account_name='sa'

    storage_url=f"s3://{bucket_name}/{model_name}/{model_version}"

    volume_op = dsl.VolumeOp(name="create pipeline volume",
                             resource_name="pipeline-pvc",
                             modes=["ReadWriteOnce"],
                             size="3Gi")

    op = dsl.ContainerOp(
        name='serve model',
        image=image,
        command=["python", f"/workspace/deployment/server.py"],
        arguments=[
            '--namespace', namespace, '--name',
            f'{model_name}-{model_version}-`', '--storeage_url', storage_url,
            '--runtime_version', runtime_version, '--service_account_name',
            service_account_name
        ],
        container_kwargs={'image_pull_policy': "IfNotPresent"},
        pvolumes={"/workspace": volume_op.volume})

    return op

@dsl.pipeline(name='Serving Pipeline',
              description='This is a single component Pipeline for Serving')
def serving_pipeline(
    image: str = 'benjamintanweihao/kubeflow-mnist',
    repo_url: str = 'https://github.com/benjamintanweihao/kubeflow-mnist.git',
):
    model_name = 'fmnist'
    export_bucket = 'servedmodels'
    model_version = '1611590079'

    git_clone = git_clone_op(repo_url=repo_url)

    serving_op(image=image,
               pvolume=git_clone.pvolume,
               bucket_name=export_bucket,
               model_name=model_name,
               model_version=model_version)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(serving_pipeline, 'serving-pipeline.zip')
