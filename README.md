# Cambricon CNNL-Example

CNNL-Example 提供基于寒武纪机器学习单元（Machine Learning Unit，MLU）开发高性能算子、C 接口封装的示例代码。

## 依赖条件

- 操作系统：
  - 目前只支持 Ubuntu 16.04 x86_64
- 寒武纪 MLU SDK：
  - 编译和运行时依赖 CNToolkit v2.3.2 或更高版本
- 寒武纪 MLU 驱动：
  - 运行时依赖驱动 v4.15.3 或更高版本

## 编译 CNNL-Example

- 获取 CNNL-Example 代码：

  ```sh
  git clone https://github.com/Cambricon/cnnl-example.git
  ```

- 准备 CNToolkit 环境：

  ```sh
  sudo apt-get install ./cntoolkit-x.x.x_Ubuntuxx.xx_amd64.deb
  sudo apt-get update
  sudo apt-get install cncc cnas cnbin cndrv cnrt
  export NEUWARE_HOME=/usr/local/neuware/
  ```

- 编译 CNNL-Example

  ```sh
  cd cnnl-example
  make -j4
  ```

  编译成功后在 cnnl-example 目录下生成 libcnnl_example.so。

# 运行 CNNL-Example

- 编译用于测试的可执行文件

  ```sh
  cd test
  make
  ```

  编译成功后会在当前目录生成 test_example 可执行文件。

- 执行测试脚本

  ```sh
  export LD_LIBRARY_PATH=${NEUWARE_HOME}:${PWD}/..:${PWD}/../lib
  ./run_test_example.sh
  ```

- 修改测试脚本

  参考 `run_test_example.sh` 中的说明。

## 目录文件结构

| 目录/文件      | 描述                                                                             |
| -------------- | -------------------------------------------------------------------------------- |
| lib            | 包含依赖库 libcnnl_core.so，支持 Ubuntu 16.04 x86_64 系统。                      |
| include        | 包含 libcnnl_core.so 中的数据类型描述，以及对外提供的 C 接口头文件 cnnl_core.h。 |
| kernels        | 算子代码实现，包含一元、二元算子模板供其他算子调用。                             |
| cnnl_example.h | kernels 目录中的算子对外提供的 C 接口头文件。                                    |
| test           | 调用 MLU 算子接口进行测试的样例。                                                |
