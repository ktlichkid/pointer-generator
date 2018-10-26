# Pointer-generator模型复现全攻略

## 概述
Pointer-generator模型是Stanford的Abigail See和Google Brain在2017年合作开发的Summarization模型，目前在处理Summarization类任务方面是state-of-art的模型。

该模型的特点是，使用了Pointer-generator机制来解决oov问题，以减少UNK的出现；使用了coverage机制来减少生成文段中的重复词。

不过，由于该模型训练过程较为复杂，且数据量较大，复现论文中的效果并不是十分容易。因此，在将该模型迁移到Paddle Fluid并开源以前，我们需要先重现TensorFlow的baseline，确保我们的运行方式正确无误，也厘清论文中提供的Rouge值是否可以达到，以便将来与Fluid的模型进行精度对齐和逐层对比等工作。

## 数据
论文中使用的CNN / Daily Mail的新闻数据已经预处理为二进制的bin文件。其中，训练集`train_*.bin`文件共288个，validation集`val_*.bin`文件共14个，测试集`test_*.bin`文件共12个。完整词典文件`vocab`中共包含200000个词条。不同于之前的问题生成模型，数据中并不包含预先训好的embedding数据。

运行前请务必核对这些数字准确无误。

## 训练流程
### 训练及Evaluation概况和原理
如果一开始就用完整的文章作为网络的input来训练，由于文章较长，那么训练速度会过于慢。
因此作者在整个训练过程中将文章长度截断在400词，并且将生成的摘要长度限制在100词（在最后预测阶段，这一长度为120词）。
同时为了进一步提高效率，在训练的早期，作者会将文章截到更短，并在训练过程中逐渐放宽文章长度限制，直到最终达到400词。
作者认为这样做不光会加速训练，还会让最终结果变好。因此，我们要复现这一结果，也需要重复作者使用的这一训练过程。

此外，coverage机制是在无coverage的模型训练到收敛后才加入。需要先将模型训练到收敛，再开启coverage机制。
这样做的目的作者并未说明，但我认为这是因为重复的问题一般多出现于训练好的模型中，因此先训到收敛再开启coverage会比较有效。
coverage机制的加入会引入新的可训练参数。如果要用Paddle Fluid来实现这样的训练流程，可能需要在模型存取方面做一些扩展。

我们在训练过程中可能会需要知道模型在validation集上的效果。但训练过程中程序不会自动计算validation集上的loss. 
因此，要做validation的话，需要在训练的同时，另启动一个任务来做validation，
这个任务会每步读取训练任务新生成的checkpoint，计算其在validation集上的loss，并且不断将其中较好的模型保存下来。

具体的训练流程见下一小节。

### 训练的具体操作流程
要复现作者在Github上提供的模型，需要按照如下流程训练：

#### 1. 启动训练和Evaluation：
```sh
python run_summarization.py --mode=train --max_enc_steps=10 --max_dec_steps=10
--data_path=/path/to/chunked/train_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```
这里先将`max_enc_steps`和`max_dec_steps`设为10，也就是将文章截得很短来训练。

这里用`--mode=train`来指定是训练。
`--data_path`参数用来指定数据文件，可以用`*`通配符来指定多个文件，这里就是指定`train_*.bin`文件们。
`--vocab_path`参数用来指定字典文件，即`vocab`文件。`--log_root`指定checkpoint文件的保存路径。`--exp_name`指定此次任务的名称。
这些参数是每次运行时都会需要指定的，接下来将不再赘述。

然后启动Evaluation任务：
```sh
python run_summarization.py --max_enc_steps=400 --max_dec_steps=100
--mode=eval --data_path=/path/to/chunked/val_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```
注意这里`max_enc_steps`和`max_dec_steps`的值为400和100，也就是说validation时不改变input文段的长度。
另外，用`--data_path`指定的数据文件是`val_*.bin`，即validation数据集。用`--mode=eval`来指定这是evaluation的任务。
其他参数需要与训练脚本的运行参数保持一致。

Evaluation任务启动后无需停止，一直运行下去就可以了，它会自动将最优模型保存下来。

#### 2. 逐步增大`max_enc_steps`和`max_dec_steps`
在训练到约71k步后，重启训练任务，修改参数为`--max_enc_steps=50 --max_dec_steps=50`，其余不变。

在训练到约116k步后，重启训练任务，修改参数为`--max_enc_steps=100 --max_dec_steps=50`，其余不变。

在训练到约184k步后，重启训练任务，修改参数为`--max_enc_steps=200 --max_dec_steps=50`，其余不变。

在训练到约223k步后，重启训练任务，修改参数为`--max_enc_steps=400 --max_dec_steps=100`，其余不变。

可以准备一个脚本来进行上述操作。

#### 3. 启动coverage
在训练到约237k步后，停止训练任务，并执行
```sh
python run_summarization.py --mode=train --max_enc_steps=400 --max_dec_steps=100 --coverage=1 --convert_to_coverage_model=1
--data_path=/path/to/chunked/train_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```
也就是将`--coverage`和`--convert_to_coverage_model`参数设为`True`，这样程序会读取之前训好的无coverage的模型，然后将模型转为coverage的模型。

执行完上述命令后，重启训练，这时将`--coverage`设为`True`，`--convert_to_coverage_model`设为`False`
```sh
python run_summarization.py --mode=train --max_enc_steps=400 --max_dec_steps=100 --coverage=1
--data_path=/path/to/chunked/train_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```
再训练约3k步，coverage loss降到0.2左右，且Evaluation的loss大致稳定，就可以停止训练了。

## 预测流程和Rouge值计算
训练完成后，用下面的命令做inference
```sh
python run_summarization.py --mode=decode --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1
--data_path=/path/to/data/test_* --vocab_path=/path/to/data/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```
这里需要将`--mode`设为`decode`，同时要将`single_pass`设为`True`，这样会将预测结果以`txt`文件形式存到log目录下。
注意`data_path`需要指定test数据集即`test_*.bin`文件。

之后可以用`rouge_cal.py`脚本来计算Rouge值。最终可以得到Rouge1 0.38，Rouge2 0.16，RougeL 0.34左右。

## 对照试验的结果
除了作者的训练方法外，我也尝试了一开始就使用较长的文章来训练。
具体而言是一开始就将`max_enc_steps`设为200，将`max_dec_steps`设为50，在训练到约100k步后增大这两个值到400，100. 
之后在230k步左右加入coverage.

实际上不同于作者所说，这样的结果要略好于作者所采用的训练方法，有Rouge1 0.39, Rouge2 0.16, RougeL 0.35左右。
因此对齐baseline时一定要注意选用完全相同的训练步骤。
