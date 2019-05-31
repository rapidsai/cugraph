/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 /*
 * debug_help.h
 *
 *  Created on: Jul 19, 2018
 *      Author: jwyles
 */

#include <string>
#include <iostream>

#pragma once

namespace debug {
	template <typename T>
	void printDeviceVector(T* dev_ptr, int items, std::string title) {
		T* host_ptr = (T*)malloc(sizeof(T) * items);
		cudaMemcpy(host_ptr, dev_ptr, sizeof(T) * items, cudaMemcpyDefault);
		std::cout << title << ": { ";
		for (int i = 0; i < items; i++) {
			std::cout << host_ptr[i] << ((i < items - 1) ? ", " : " ");
		}
		std::cout << "}\n";
		free(host_ptr);
	}
}
