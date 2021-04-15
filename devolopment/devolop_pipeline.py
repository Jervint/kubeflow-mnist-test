# -*- encoding: utf-8 -*-
'''
Date: 2021-04-14 17:20:50
LastEditors: Jervint
LastEditTime: 2021-04-15 18:04:42
Description: 
FilePath: /kubeflow-pytorch-test/devolopment/devolop_pipeline.py
'''
import kfp.dsl as dsl
from kfp.dsl import PipelineVolume


def preprocess_op(docker_image_path: str, dataset_root_dir: str,
                  dataset_name: str):
    def volume_op():
        return dsl.VolumeOp(name="create pipeline volume",
                            resource_name="pipeline-pvc",
                            modes=["ReadWriteOnce"],
                            size="3Gi")

    pvolume = volume_op()
    return dsl.ContainerOp(
        name="preprocessing",
        image=docker_image_path,
        command=["python", f"preprocess.py"],
        arguments=[
            "--dataset_root_dir", dataset_root_dir, "--dataset_name",
            dataset_name
        ],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume.volume})


def train_op(docker_image_path: str, pvolume: PipelineVolume,
             dataset_root_dir: str, dataset_name: str):
    return dsl.ContainerOp(
        name="train",
        image=docker_image_path,
        command=["python", f"train.py"],
        arguments=[
            "--dataset_root_dir", dataset_root_dir, "--dataset_name",
            dataset_name
        ],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume})


@dsl.pipeline(
    name='Fashion MNIST Training Pytorch Pipeline',
    description=
    'Fashion MNIST Training Pipeline to be executed on KubeFlow-Pytorch.')
def train_pipeline(
        image: str = "harbor.qunhequnhe.com/koolab/kubeflow-mnist:train",
        dataset_root_dir: str = "/workspace/datasets",
        dataset_name: str = "fashion-mnist"):

    preprocess_data = preprocess_op(image,
                                    dataset_root_dir=dataset_root_dir,
                                    dataset_name=dataset_name)
    train_and_eval = train_op(image,
                              pvolume=preprocess_data.pvolume,
                              dataset_root_dir=dataset_root_dir,
                              dataset_name=dataset_name)


if __name__ == "__main__":
    import kfp.compiler as compiler

    compiler.Compiler().compile(training_pipeline, __file__ + '.tar.gz')