## WholeMemory
WholeMemory can be regarded as a whole view of GPU memory.
WholeMemory exposes a handle to the memory instance no matter how the underlying data is stored across multiple GPUs.
WholeMemory assumes that a separate process is used to control each GPU.

### WholeMemory Basics
To define WholeMemory, we need to specify the following:

#### 1. Specify the set of GPU to handle the Memory

As WholeMemory is owned by a set of GPUs, so the set of GPUs need to be specified.
This is done by creating [WholeMemory Communicator](#wholememory-communicator) and specify the WholeMemory Communicator
when creating WholeMemory.

#### 2. Specify the location of the memory

Although WholeMemory is owned by a set of GPUs,  the memory itself can be located on host memory or on device memory.
So the location of the memory needs to be specified. Two types of location can be specified.

- **Host memory**: will use pinned host memory as underlying storage.
- **Device memory**: will use GPU device memory as underlying storage.

#### 3. Specify the address mapping mode of the memory

As WholeMemory is owned by multiple GPUs, each GPU will access the whole memory space, so we need address mapping.
There are three types of address mapping modes (also known as WholeMemory types), they are:

- **Continuous**: All memory from each GPU will be mapped into a single continuous memory address space for each GPU.
  In this mode, each GPU can directly access the whole memory using a single pointer and offset, just like using normal
  device memory. Software will see no difference. Hardware peer-to-peer access will handle the underlying communication.

- **Chunked**: Memory from each GPU will be mapped into different memory chunks, one chunk for each GPU.
  In this mode, direct access is also supported, but not using a single pointer. Software will see the chunked memory.
  However, an abstract layer can hide this.

- **Distributed**: Memory from other GPUs is not mapped into current GPU, so no direct access is supported.
  To access memory of another GPU, explicit communication is needed.

If you would like to know more details about WholeMemory locations and WholeMemory types, please refer to
[WholeMemory Implementation Details](wholememory_implementation_details.md)

### WholeMemory Communicator
WholeMemory Communicator has two main purpose:

- **Defines a set of GPUs which works together on WholeMemory.** WholeMemory Communicator is created by all GPUs that
  wants to work together. A WholeMemory Communicator can be reused as long as the GPU set needed is the same.
- **Provides underlying communication channel needed by WholeMemory.** WholeMemory may need commuincator between GPUs
  during the WholeMemory creation and some OPs on some types of WholeMemory.

To Create WholeMemory Communicator, a WholeMemory Unique ID need to be created first, it is usually created by the first
GPU in the set of GPUs, and then broadcasted to all GPUs that want to work together. Then all GPUs in this communicator
will call WholeMemory Communicator creation function using this WholeMemory Unique ID, and the rank of current GPU as
well as all GPU count.

### WholeMemory Granularity
As underlying storage may be physically partitioned into multiple GPUs, it is usually not wanted inside one single
user data block. To help with this, when creating WholeMemory, the granularity of data can be specified. Therefore
WholeMemory is considered as multiple blocks of the same granularity and will not get split inside the granularity.

### WholeMemory Mapping
Since WholeMemory provides a whole view of memory to GPU,  mapping is usually needed to access WholeMemory.
Different types of WholeMemory have different mapping methods supported as their names.
Some mappings supported include:
- All the WholeMemory types support mapping the memory range that local GPU is responsible for.
  That is, each rank can directly access "Local" memory in all types of WholeMemory.
  Here "Local" memory doesn't have to be on current GPU's memory, it can be on host memory or even maybe on other GPU,
  but it is guaranteed to be directly accessed by current GPU.
- Chunked and Continuous WholeMemory also support Chunked mapping. That is, memory of all GPUs can be mapped into
  current GPU, one continuous chunk for one GPU. Each chunk can be directly accessed by current GPU. But the memory of
  different chunks are not guaranteed to be continuous.
- Continuous WholeMemory can be mapped into continuous memory space. That is, memory of all GPUs are mapped into a
  single range of virtual memory, accessing different positions of this memory will physically access different
  GPUs. This mapping will be handled by hardware (CPU pagetable or GPU pagetable).

### Operations on WholeMemory
There are some operations that can be performed on WholeMemory. They are based on the mapping of WholeMemory.
#### Local Operation
As all WholeMemory supports mapping of local memory, so operation on local memory is supported. The operation can be
either read or write. Just use it as GPU memory of current device is OK.
#### Load / Store
To facilitate file operation, Load / Store WholeMemory from file or to file is supported. WholeMemory use raw binary
file format for disk operation. For Load, the input file can be single file or a list of files, if it is a list, they
will be logically concatenated together and then loaded. For store, each GPU stores its local memory to file, producing
a list of files.
#### Gather / Scatter
WholeMemory also supports Gather / Scatter operations, usually they operate on a
[WholeMemory Tensor](#wholememory-tensor).

### WholeMemory Tensor
Compared to PyTorch, WholeMemory is like PyTorch Storage while WholeMemory Tensor is like PyTorch Tensor.
For now, WholeMemory supports only 1D and 2D tensor, or array and matrix. Only first dimension is partitioned.

### WholeMemory Embedding
WholeMemory Embedding is just like 2D WholeMemory Tensor, with  cache support and sparse optimizer support added.
#### Cache Support
WholeMemory Embedding supports cache. To create WholeMemory Embedding with cache, WholeMemory CachePolicy need first be
created. WholeMemoryCachePolicy can be created with following fields:
- **WholeMemory Communicator**: WholeMemory CachePolicy also need WholeMemory Communicator.
  This WholeMemory Communicator defines the set of GPUs that cache the all the Embedding.
  It can be the same as the WholeMemory Communicator used to create WholeMemory Embedding.
- **WholeMemory type**: WholeMemory CachePolicy uses WholeMemory type to specify the WholeMemory type of the cache.
- **WholeMemory location**: WholeMemory CachePolicy uses WholeMemory location to specify the location of the cache.
- **Access type**: Access type can be readonly or readwrite.
- **Cache ratio**: Specify how much memory the cache will use. This ratio is computed for each GPU set that caches the
  whole embedding.

There are two most commonly used caches. They are:
- **Device cached host memory**: When the WholeMemory Communicator for Cache Policy is the same as the WholeMemory
  Communicator used to create WholeMemory Embedding, it means that cache has the same GPU set as WholeMemory Embedding.
  So each GPU just cache its own part of raw Embedding.
  Normally, when raw WholeMemory Embedding is located on host memory, and the cache is on device
  memory, each GPU just caches its own part of host memory.
- **Local cached global memory**: The WholeMemory Communicator of WholeMemory CachePolicy can also be a subset of the
  WholeMemory Communicator of WholeMemory Embedding. In this case, the subset of GPUs together cache all the embeddings.
  Typically, raw WholeMemory Embedding is partitioned on different machine nodes, and we
  want to cache some embeddings in local machine or local GPU, then the subset of GPUs can be all the GPUs on the local
  machine. For local cached global memory supports just readonly.

#### WholeMemory Embedding Sparse Optimizer
Another feature of WholeMemory Embedding is that WholeMemory Embedding supports embedding training.
To efficiently train large embedding tables, a sparse optimizer is needed.
The WholeMemory Embedding Sparse Optimizer can run on cached or non-cached WholeMemory Embedding.
Currently supported optimizers include SGD, Adam, RMSProp and AdaGrad.
