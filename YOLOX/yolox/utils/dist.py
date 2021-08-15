#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This file mainly comes from
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/comm.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import functools
import logging
import pickle
import time
import numpy as np

import oneflow as flow


__all__ = [
    "is_main_process",
    "time_synchronized",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "get_local_size"
]

_LOCAL_PROCESS_GROUP = None

def is_main_process():
    return flow._oneflow_internal.GetRank() == 0

def time_synchronized():
    """pytorch-accurate time"""
    return time.time()

def get_local_rank():
    return flow._oneflow_internal.GetLocalRank()

def get_local_size():
    return len(flow._oneflow_internal.GetLocalRank())

def get_rank():
    """Returns the rank of current process group.

    Returns:
        The rank of the process group.

    """
    return flow._oneflow_internal.GetRank()


def get_world_size():
    """Returns the number of processes in the current process group.

    Returns:
        The world size of the process group.

    """
    return flow._oneflow_internal.GetWorldSize()
