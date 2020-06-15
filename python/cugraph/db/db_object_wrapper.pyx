# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# distutils: language = c++

from db_object cimport db_object

cdef class db_big:
    cdef db_object[long]*c_db

    def __cinit__(self):
        self.c_db = new db_object[long]()

    def __dealloc__(self):
        del self.c_db

    def query(self):
        return self.c_db.query()

    def toString(self):
        return self.c_db.toString()

    def __str__(self):
        return self.c_db.toString()

cdef class db_small:
    cdef db_object[int]*c_db

    def __cinit__(self):
        self.c_db = new db_object[int]()

    def __dealloc__(self):
        del self.c_db

    def query(self):
        return self.c_db.query()

    def toString(self):
        return self.c_db.toString()

    def __str__(self):
        return self.c_db.toString()