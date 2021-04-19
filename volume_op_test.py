# -*- encoding: utf-8 -*-
'''
Date: 2021-04-14 17:20:50
LastEditors: Jervint
LastEditTime: 2021-04-19 09:42:28
Description: 
FilePath: /kubeflow-pytorch-test/volume_op_test.py
'''
import kfp.dsl as dsl
from kfp.dsl import PipelineVolume
from kubernetes.client.models import V1EnvVar


def volume_op():
    return dsl.VolumeOp(name="create pipeline volume",
                        resource_name="pipeline-pvc",
                        modes=["ReadWriteOnce"],
                        size="1Gi")


def preprocess_op(docker_image_path: str, pvolume: PipelineVolume,
                  dataset_root_dir: str, dataset_name: str):
    return dsl.ContainerOp(
        name="preprocessing",
        image=docker_image_path,
        command=["python", f"/myworkspace/devolopment/preprocess.py"],
        arguments=[
            "--dataset_root_dir", dataset_root_dir, "--dataset_name",
            dataset_name, '--train_batch_size', 64, '--test_batch_size', 64,
            '--num_workers', 4
        ],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/myworkspace": pvolume})


def train_op(docker_image_path: str, pvolume: PipelineVolume,
             dataset_root_dir: str, dataset_name: str, num_gpu: int,
             gpu_list: str):
    # num_gpus = len(gpu_list.value.split(','))
    # gpu_list = ""
    # for i in range(num_gpu):
    #     gpu_list += (str(i) + ",")
    env_var = V1EnvVar(name="CUDA_VISIBLE_DEVICES", value="0,1")
    return dsl.ContainerOp(
        name="train",
        image=docker_image_path,
        command=["python", f"/myworkspace/devolopment/train.py"],
        arguments=[
            "--dataset_root_dir", dataset_root_dir, "--dataset_name",
            dataset_name
        ],
        file_outputs={
            'output': f'/myworkspace/devolopment/output.txt'
        },
        container_kwargs={
            'image_pull_policy': 'IfNotPresent'
        },
        pvolumes={
            "/myworkspace": pvolume
        }).container.set_gpu_limit(2).add_env_variable(env_var)


@dsl.pipeline(
    name='Fashion MNIST Training Pytorch Pipeline',
    description=
    'Fashion MNIST Training Pipeline to be executed on KubeFlow-Pytorch.')
def train_pipeline(
        # image: str = "harbor.qunhequnhe.com/koolab/kubeflow-mnist:v1",
        image: str = "",
        dataset_root_dir: str = "/myworkspace/datasets",
        dataset_name: str = "fashion-mnist",
        num_gpu: int = 1,
        gpu_list: str = "0"):
    volume_ = volume_op()
    preprocess_op=


if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(training_pipeline, __file__ + '.tar.gz')