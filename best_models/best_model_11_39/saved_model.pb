��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.12unknown8��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
v
dense_3568/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3568/bias
o
#dense_3568/bias/Read/ReadVariableOpReadVariableOpdense_3568/bias*
_output_shapes
:*
dtype0
~
dense_3568/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*"
shared_namedense_3568/kernel
w
%dense_3568/kernel/Read/ReadVariableOpReadVariableOpdense_3568/kernel*
_output_shapes

:@*
dtype0
v
dense_3567/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_namedense_3567/bias
o
#dense_3567/bias/Read/ReadVariableOpReadVariableOpdense_3567/bias*
_output_shapes
:@*
dtype0
~
dense_3567/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*"
shared_namedense_3567/kernel
w
%dense_3567/kernel/Read/ReadVariableOpReadVariableOpdense_3567/kernel*
_output_shapes

:@*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
#_self_saveable_object_factories*
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias
#&_self_saveable_object_factories*
 
0
1
$2
%3*
 
0
1
$2
%3*
* 
�
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
,trace_0
-trace_1
.trace_2
/trace_3* 
6
0trace_0
1trace_1
2trace_2
3trace_3* 
* 
* 

4serving_default* 
* 
* 
* 
* 
�
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

:trace_0* 

;trace_0* 
* 

0
1*

0
1*
* 
�
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Atrace_0* 

Btrace_0* 
a[
VARIABLE_VALUEdense_3567/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_3567/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

$0
%1*

$0
%1*
* 
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

Htrace_0* 

Itrace_0* 
a[
VARIABLE_VALUEdense_3568/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_3568/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

0
1
2*

J0
K1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
L	variables
M	keras_api
	Ntotal
	Ocount*
H
P	variables
Q	keras_api
	Rtotal
	Scount
T
_fn_kwargs*

N0
O1*

L	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

R0
S1*

P	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
!serving_default_flatten_351_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall!serving_default_flatten_351_inputdense_3567/kerneldense_3567/biasdense_3568/kerneldense_3568/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_473605
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%dense_3567/kernel/Read/ReadVariableOp#dense_3567/bias/Read/ReadVariableOp%dense_3568/kernel/Read/ReadVariableOp#dense_3568/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_473769
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3567/kerneldense_3567/biasdense_3568/kerneldense_3568/biastotal_1count_1totalcount*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_473803̼
�

�
F__inference_dense_3567_layer_call_and_return_conditional_losses_473445

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_sequential_533_layer_call_and_return_conditional_losses_473575
flatten_351_input#
dense_3567_473564:@
dense_3567_473566:@#
dense_3568_473569:@
dense_3568_473571:
identity��"dense_3567/StatefulPartitionedCall�"dense_3568/StatefulPartitionedCall�
flatten_351/PartitionedCallPartitionedCallflatten_351_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_351_layer_call_and_return_conditional_losses_473432�
"dense_3567/StatefulPartitionedCallStatefulPartitionedCall$flatten_351/PartitionedCall:output:0dense_3567_473564dense_3567_473566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_3567_layer_call_and_return_conditional_losses_473445�
"dense_3568/StatefulPartitionedCallStatefulPartitionedCall+dense_3567/StatefulPartitionedCall:output:0dense_3568_473569dense_3568_473571*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_3568_layer_call_and_return_conditional_losses_473462z
IdentityIdentity+dense_3568/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^dense_3567/StatefulPartitionedCall#^dense_3568/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 2H
"dense_3567/StatefulPartitionedCall"dense_3567/StatefulPartitionedCall2H
"dense_3568/StatefulPartitionedCall"dense_3568/StatefulPartitionedCall:^ Z
+
_output_shapes
:���������
+
_user_specified_nameflatten_351_input
�
�
J__inference_sequential_533_layer_call_and_return_conditional_losses_473590
flatten_351_input#
dense_3567_473579:@
dense_3567_473581:@#
dense_3568_473584:@
dense_3568_473586:
identity��"dense_3567/StatefulPartitionedCall�"dense_3568/StatefulPartitionedCall�
flatten_351/PartitionedCallPartitionedCallflatten_351_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_351_layer_call_and_return_conditional_losses_473432�
"dense_3567/StatefulPartitionedCallStatefulPartitionedCall$flatten_351/PartitionedCall:output:0dense_3567_473579dense_3567_473581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_3567_layer_call_and_return_conditional_losses_473445�
"dense_3568/StatefulPartitionedCallStatefulPartitionedCall+dense_3567/StatefulPartitionedCall:output:0dense_3568_473584dense_3568_473586*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_3568_layer_call_and_return_conditional_losses_473462z
IdentityIdentity+dense_3568/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^dense_3567/StatefulPartitionedCall#^dense_3568/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 2H
"dense_3567/StatefulPartitionedCall"dense_3567/StatefulPartitionedCall2H
"dense_3568/StatefulPartitionedCall"dense_3568/StatefulPartitionedCall:^ Z
+
_output_shapes
:���������
+
_user_specified_nameflatten_351_input
�
�
!__inference__wrapped_model_473419
flatten_351_inputJ
8sequential_533_dense_3567_matmul_readvariableop_resource:@G
9sequential_533_dense_3567_biasadd_readvariableop_resource:@J
8sequential_533_dense_3568_matmul_readvariableop_resource:@G
9sequential_533_dense_3568_biasadd_readvariableop_resource:
identity��0sequential_533/dense_3567/BiasAdd/ReadVariableOp�/sequential_533/dense_3567/MatMul/ReadVariableOp�0sequential_533/dense_3568/BiasAdd/ReadVariableOp�/sequential_533/dense_3568/MatMul/ReadVariableOpq
 sequential_533/flatten_351/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
"sequential_533/flatten_351/ReshapeReshapeflatten_351_input)sequential_533/flatten_351/Const:output:0*
T0*'
_output_shapes
:����������
/sequential_533/dense_3567/MatMul/ReadVariableOpReadVariableOp8sequential_533_dense_3567_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
 sequential_533/dense_3567/MatMulMatMul+sequential_533/flatten_351/Reshape:output:07sequential_533/dense_3567/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0sequential_533/dense_3567/BiasAdd/ReadVariableOpReadVariableOp9sequential_533_dense_3567_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
!sequential_533/dense_3567/BiasAddBiasAdd*sequential_533/dense_3567/MatMul:product:08sequential_533/dense_3567/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
sequential_533/dense_3567/ReluRelu*sequential_533/dense_3567/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
/sequential_533/dense_3568/MatMul/ReadVariableOpReadVariableOp8sequential_533_dense_3568_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
 sequential_533/dense_3568/MatMulMatMul,sequential_533/dense_3567/Relu:activations:07sequential_533/dense_3568/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
0sequential_533/dense_3568/BiasAdd/ReadVariableOpReadVariableOp9sequential_533_dense_3568_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!sequential_533/dense_3568/BiasAddBiasAdd*sequential_533/dense_3568/MatMul:product:08sequential_533/dense_3568/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!sequential_533/dense_3568/SoftmaxSoftmax*sequential_533/dense_3568/BiasAdd:output:0*
T0*'
_output_shapes
:���������z
IdentityIdentity+sequential_533/dense_3568/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp1^sequential_533/dense_3567/BiasAdd/ReadVariableOp0^sequential_533/dense_3567/MatMul/ReadVariableOp1^sequential_533/dense_3568/BiasAdd/ReadVariableOp0^sequential_533/dense_3568/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 2d
0sequential_533/dense_3567/BiasAdd/ReadVariableOp0sequential_533/dense_3567/BiasAdd/ReadVariableOp2b
/sequential_533/dense_3567/MatMul/ReadVariableOp/sequential_533/dense_3567/MatMul/ReadVariableOp2d
0sequential_533/dense_3568/BiasAdd/ReadVariableOp0sequential_533/dense_3568/BiasAdd/ReadVariableOp2b
/sequential_533/dense_3568/MatMul/ReadVariableOp/sequential_533/dense_3568/MatMul/ReadVariableOp:^ Z
+
_output_shapes
:���������
+
_user_specified_nameflatten_351_input
�
�
/__inference_sequential_533_layer_call_fn_473560
flatten_351_input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallflatten_351_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_533_layer_call_and_return_conditional_losses_473536o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
+
_output_shapes
:���������
+
_user_specified_nameflatten_351_input
�
�
+__inference_dense_3567_layer_call_fn_473691

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_3567_layer_call_and_return_conditional_losses_473445o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_dense_3568_layer_call_fn_473711

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_3568_layer_call_and_return_conditional_losses_473462o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
J__inference_sequential_533_layer_call_and_return_conditional_losses_473469

inputs#
dense_3567_473446:@
dense_3567_473448:@#
dense_3568_473463:@
dense_3568_473465:
identity��"dense_3567/StatefulPartitionedCall�"dense_3568/StatefulPartitionedCall�
flatten_351/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_351_layer_call_and_return_conditional_losses_473432�
"dense_3567/StatefulPartitionedCallStatefulPartitionedCall$flatten_351/PartitionedCall:output:0dense_3567_473446dense_3567_473448*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_3567_layer_call_and_return_conditional_losses_473445�
"dense_3568/StatefulPartitionedCallStatefulPartitionedCall+dense_3567/StatefulPartitionedCall:output:0dense_3568_473463dense_3568_473465*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_3568_layer_call_and_return_conditional_losses_473462z
IdentityIdentity+dense_3568/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^dense_3567/StatefulPartitionedCall#^dense_3568/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 2H
"dense_3567/StatefulPartitionedCall"dense_3567/StatefulPartitionedCall2H
"dense_3568/StatefulPartitionedCall"dense_3568/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
F__inference_dense_3568_layer_call_and_return_conditional_losses_473462

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
/__inference_sequential_533_layer_call_fn_473631

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_533_layer_call_and_return_conditional_losses_473536o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference__traced_save_473769
file_prefix0
,savev2_dense_3567_kernel_read_readvariableop.
*savev2_dense_3567_bias_read_readvariableop0
,savev2_dense_3568_kernel_read_readvariableop.
*savev2_dense_3568_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_dense_3567_kernel_read_readvariableop*savev2_dense_3567_bias_read_readvariableop,savev2_dense_3568_kernel_read_readvariableop*savev2_dense_3568_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes.
,: :@:@:@:: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�
�
$__inference_signature_wrapper_473605
flatten_351_input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallflatten_351_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_473419o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
+
_output_shapes
:���������
+
_user_specified_nameflatten_351_input
�#
�
"__inference__traced_restore_473803
file_prefix4
"assignvariableop_dense_3567_kernel:@0
"assignvariableop_1_dense_3567_bias:@6
$assignvariableop_2_dense_3568_kernel:@0
"assignvariableop_3_dense_3568_bias:$
assignvariableop_4_total_1: $
assignvariableop_5_count_1: "
assignvariableop_6_total: "
assignvariableop_7_count: 

identity_9��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp"assignvariableop_dense_3567_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_3567_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_3568_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_3568_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_total_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_count_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_totalIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_countIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_9IdentityIdentity_8:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*"
_acd_function_control_output(*
_output_shapes
 "!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
/__inference_sequential_533_layer_call_fn_473618

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_533_layer_call_and_return_conditional_losses_473469o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
G__inference_flatten_351_layer_call_and_return_conditional_losses_473682

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
F__inference_dense_3568_layer_call_and_return_conditional_losses_473722

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
J__inference_sequential_533_layer_call_and_return_conditional_losses_473536

inputs#
dense_3567_473525:@
dense_3567_473527:@#
dense_3568_473530:@
dense_3568_473532:
identity��"dense_3567/StatefulPartitionedCall�"dense_3568/StatefulPartitionedCall�
flatten_351/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_351_layer_call_and_return_conditional_losses_473432�
"dense_3567/StatefulPartitionedCallStatefulPartitionedCall$flatten_351/PartitionedCall:output:0dense_3567_473525dense_3567_473527*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_3567_layer_call_and_return_conditional_losses_473445�
"dense_3568/StatefulPartitionedCallStatefulPartitionedCall+dense_3567/StatefulPartitionedCall:output:0dense_3568_473530dense_3568_473532*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_3568_layer_call_and_return_conditional_losses_473462z
IdentityIdentity+dense_3568/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^dense_3567/StatefulPartitionedCall#^dense_3568/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 2H
"dense_3567/StatefulPartitionedCall"dense_3567/StatefulPartitionedCall2H
"dense_3568/StatefulPartitionedCall"dense_3568/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_sequential_533_layer_call_and_return_conditional_losses_473671

inputs;
)dense_3567_matmul_readvariableop_resource:@8
*dense_3567_biasadd_readvariableop_resource:@;
)dense_3568_matmul_readvariableop_resource:@8
*dense_3568_biasadd_readvariableop_resource:
identity��!dense_3567/BiasAdd/ReadVariableOp� dense_3567/MatMul/ReadVariableOp�!dense_3568/BiasAdd/ReadVariableOp� dense_3568/MatMul/ReadVariableOpb
flatten_351/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   t
flatten_351/ReshapeReshapeinputsflatten_351/Const:output:0*
T0*'
_output_shapes
:����������
 dense_3567/MatMul/ReadVariableOpReadVariableOp)dense_3567_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_3567/MatMulMatMulflatten_351/Reshape:output:0(dense_3567/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
!dense_3567/BiasAdd/ReadVariableOpReadVariableOp*dense_3567_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_3567/BiasAddBiasAdddense_3567/MatMul:product:0)dense_3567/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@f
dense_3567/ReluReludense_3567/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
 dense_3568/MatMul/ReadVariableOpReadVariableOp)dense_3568_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_3568/MatMulMatMuldense_3567/Relu:activations:0(dense_3568/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_3568/BiasAdd/ReadVariableOpReadVariableOp*dense_3568_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3568/BiasAddBiasAdddense_3568/MatMul:product:0)dense_3568/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
dense_3568/SoftmaxSoftmaxdense_3568/BiasAdd:output:0*
T0*'
_output_shapes
:���������k
IdentityIdentitydense_3568/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_3567/BiasAdd/ReadVariableOp!^dense_3567/MatMul/ReadVariableOp"^dense_3568/BiasAdd/ReadVariableOp!^dense_3568/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 2F
!dense_3567/BiasAdd/ReadVariableOp!dense_3567/BiasAdd/ReadVariableOp2D
 dense_3567/MatMul/ReadVariableOp dense_3567/MatMul/ReadVariableOp2F
!dense_3568/BiasAdd/ReadVariableOp!dense_3568/BiasAdd/ReadVariableOp2D
 dense_3568/MatMul/ReadVariableOp dense_3568/MatMul/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_sequential_533_layer_call_and_return_conditional_losses_473651

inputs;
)dense_3567_matmul_readvariableop_resource:@8
*dense_3567_biasadd_readvariableop_resource:@;
)dense_3568_matmul_readvariableop_resource:@8
*dense_3568_biasadd_readvariableop_resource:
identity��!dense_3567/BiasAdd/ReadVariableOp� dense_3567/MatMul/ReadVariableOp�!dense_3568/BiasAdd/ReadVariableOp� dense_3568/MatMul/ReadVariableOpb
flatten_351/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   t
flatten_351/ReshapeReshapeinputsflatten_351/Const:output:0*
T0*'
_output_shapes
:����������
 dense_3567/MatMul/ReadVariableOpReadVariableOp)dense_3567_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_3567/MatMulMatMulflatten_351/Reshape:output:0(dense_3567/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
!dense_3567/BiasAdd/ReadVariableOpReadVariableOp*dense_3567_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_3567/BiasAddBiasAdddense_3567/MatMul:product:0)dense_3567/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@f
dense_3567/ReluReludense_3567/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
 dense_3568/MatMul/ReadVariableOpReadVariableOp)dense_3568_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_3568/MatMulMatMuldense_3567/Relu:activations:0(dense_3568/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_3568/BiasAdd/ReadVariableOpReadVariableOp*dense_3568_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3568/BiasAddBiasAdddense_3568/MatMul:product:0)dense_3568/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
dense_3568/SoftmaxSoftmaxdense_3568/BiasAdd:output:0*
T0*'
_output_shapes
:���������k
IdentityIdentitydense_3568/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_3567/BiasAdd/ReadVariableOp!^dense_3567/MatMul/ReadVariableOp"^dense_3568/BiasAdd/ReadVariableOp!^dense_3568/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 2F
!dense_3567/BiasAdd/ReadVariableOp!dense_3567/BiasAdd/ReadVariableOp2D
 dense_3567/MatMul/ReadVariableOp dense_3567/MatMul/ReadVariableOp2F
!dense_3568/BiasAdd/ReadVariableOp!dense_3568/BiasAdd/ReadVariableOp2D
 dense_3568/MatMul/ReadVariableOp dense_3568/MatMul/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
F__inference_dense_3567_layer_call_and_return_conditional_losses_473702

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_sequential_533_layer_call_fn_473480
flatten_351_input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallflatten_351_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_533_layer_call_and_return_conditional_losses_473469o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
+
_output_shapes
:���������
+
_user_specified_nameflatten_351_input
�
H
,__inference_flatten_351_layer_call_fn_473676

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_351_layer_call_and_return_conditional_losses_473432`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
G__inference_flatten_351_layer_call_and_return_conditional_losses_473432

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
S
flatten_351_input>
#serving_default_flatten_351_input:0���������>

dense_35680
StatefulPartitionedCall:0���������tensorflow/serving/predict:�k
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
#_self_saveable_object_factories"
_tf_keras_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias
#&_self_saveable_object_factories"
_tf_keras_layer
<
0
1
$2
%3"
trackable_list_wrapper
<
0
1
$2
%3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
�
,trace_0
-trace_1
.trace_2
/trace_32�
/__inference_sequential_533_layer_call_fn_473480
/__inference_sequential_533_layer_call_fn_473618
/__inference_sequential_533_layer_call_fn_473631
/__inference_sequential_533_layer_call_fn_473560�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z,trace_0z-trace_1z.trace_2z/trace_3
�
0trace_0
1trace_1
2trace_2
3trace_32�
J__inference_sequential_533_layer_call_and_return_conditional_losses_473651
J__inference_sequential_533_layer_call_and_return_conditional_losses_473671
J__inference_sequential_533_layer_call_and_return_conditional_losses_473575
J__inference_sequential_533_layer_call_and_return_conditional_losses_473590�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z0trace_0z1trace_1z2trace_2z3trace_3
�B�
!__inference__wrapped_model_473419flatten_351_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
	optimizer
,
4serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
:trace_02�
,__inference_flatten_351_layer_call_fn_473676�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z:trace_0
�
;trace_02�
G__inference_flatten_351_layer_call_and_return_conditional_losses_473682�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z;trace_0
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Atrace_02�
+__inference_dense_3567_layer_call_fn_473691�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zAtrace_0
�
Btrace_02�
F__inference_dense_3567_layer_call_and_return_conditional_losses_473702�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zBtrace_0
#:!@2dense_3567/kernel
:@2dense_3567/bias
 "
trackable_dict_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
Htrace_02�
+__inference_dense_3568_layer_call_fn_473711�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zHtrace_0
�
Itrace_02�
F__inference_dense_3568_layer_call_and_return_conditional_losses_473722�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zItrace_0
#:!@2dense_3568/kernel
:2dense_3568/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_sequential_533_layer_call_fn_473480flatten_351_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
/__inference_sequential_533_layer_call_fn_473618inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
/__inference_sequential_533_layer_call_fn_473631inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
/__inference_sequential_533_layer_call_fn_473560flatten_351_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_sequential_533_layer_call_and_return_conditional_losses_473651inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_sequential_533_layer_call_and_return_conditional_losses_473671inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_sequential_533_layer_call_and_return_conditional_losses_473575flatten_351_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_sequential_533_layer_call_and_return_conditional_losses_473590flatten_351_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
$__inference_signature_wrapper_473605flatten_351_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_flatten_351_layer_call_fn_473676inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_flatten_351_layer_call_and_return_conditional_losses_473682inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_3567_layer_call_fn_473691inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_3567_layer_call_and_return_conditional_losses_473702inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_3568_layer_call_fn_473711inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_3568_layer_call_and_return_conditional_losses_473722inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
L	variables
M	keras_api
	Ntotal
	Ocount"
_tf_keras_metric
^
P	variables
Q	keras_api
	Rtotal
	Scount
T
_fn_kwargs"
_tf_keras_metric
.
N0
O1"
trackable_list_wrapper
-
L	variables"
_generic_user_object
:  (2total
:  (2count
.
R0
S1"
trackable_list_wrapper
-
P	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
!__inference__wrapped_model_473419$%>�;
4�1
/�,
flatten_351_input���������
� "7�4
2

dense_3568$�!

dense_3568����������
F__inference_dense_3567_layer_call_and_return_conditional_losses_473702\/�,
%�"
 �
inputs���������
� "%�"
�
0���������@
� ~
+__inference_dense_3567_layer_call_fn_473691O/�,
%�"
 �
inputs���������
� "����������@�
F__inference_dense_3568_layer_call_and_return_conditional_losses_473722\$%/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� ~
+__inference_dense_3568_layer_call_fn_473711O$%/�,
%�"
 �
inputs���������@
� "�����������
G__inference_flatten_351_layer_call_and_return_conditional_losses_473682\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������
� 
,__inference_flatten_351_layer_call_fn_473676O3�0
)�&
$�!
inputs���������
� "�����������
J__inference_sequential_533_layer_call_and_return_conditional_losses_473575u$%F�C
<�9
/�,
flatten_351_input���������
p 

 
� "%�"
�
0���������
� �
J__inference_sequential_533_layer_call_and_return_conditional_losses_473590u$%F�C
<�9
/�,
flatten_351_input���������
p

 
� "%�"
�
0���������
� �
J__inference_sequential_533_layer_call_and_return_conditional_losses_473651j$%;�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������
� �
J__inference_sequential_533_layer_call_and_return_conditional_losses_473671j$%;�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������
� �
/__inference_sequential_533_layer_call_fn_473480h$%F�C
<�9
/�,
flatten_351_input���������
p 

 
� "�����������
/__inference_sequential_533_layer_call_fn_473560h$%F�C
<�9
/�,
flatten_351_input���������
p

 
� "�����������
/__inference_sequential_533_layer_call_fn_473618]$%;�8
1�.
$�!
inputs���������
p 

 
� "�����������
/__inference_sequential_533_layer_call_fn_473631]$%;�8
1�.
$�!
inputs���������
p

 
� "�����������
$__inference_signature_wrapper_473605�$%S�P
� 
I�F
D
flatten_351_input/�,
flatten_351_input���������"7�4
2

dense_3568$�!

dense_3568���������