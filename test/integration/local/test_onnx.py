#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
from __future__ import absolute_import

import os

import local_mode_utils
import numpy
from sagemaker.mxnet import MXNetModel
from test.integration import NUM_MODEL_SERVER_WORKERS, RESOURCE_PATH

ONNX_PATH = os.path.join(RESOURCE_PATH, 'onnx')
SCRIPT_PATH = os.path.join(ONNX_PATH, 'code', 'onnx_export_import.py')


def test_onnx_import(docker_image, sagemaker_local_session, local_instance_type):
    model_path = 'file://{}'.format(os.path.join(ONNX_PATH, 'onnx_model'))
    m = MXNetModel(model_path, 'SageMakerRole', SCRIPT_PATH, image=docker_image,
                   sagemaker_session=sagemaker_local_session,
                   model_server_workers=NUM_MODEL_SERVER_WORKERS)

    input = numpy.zeros(shape=(1, 1, 28, 28))

    with local_mode_utils.lock():
        try:
            predictor = m.deploy(1, local_instance_type)
            output = predictor.predict(input)
        finally:
            sagemaker_local_session.delete_endpoint(m.endpoint_name)

    # Check that there is a probability for each possible class in the prediction
    assert len(output[0]) == 10
