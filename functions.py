import torch
import copy
import sys
import time 
import pickle
import numpy as np
import warnings
from scipy.interpolate import Rbf

from collections import OrderedDict
from constants import *

def update_progress(index, length, **kwargs): # progess 를 보여줌
    '''
        display progress
        
        Input:
            `index`: (int) shows the index of current progress # 현재 progress의 인덱스를 보여줌
            `length`: (int) total length of the progress # progress의 총 길이를 보여줌
            `**kwargs`: info to display (e.g. accuracy) # display의 정보(정확도)
    '''
    barLength = 10 # Modify this to change the length of the progress bar
    progress = float(index/length)
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.2f}% ({2}/{3}) ".format( 
            "#"*block + "-"*(barLength-block), round(progress*100, 3), \
            index, length)
    for key, value in kwargs.items():
        text = text + str(key) + ': ' + str(value) + ', '
    if len(kwargs) != 0:
        text = text[:-2:]
    sys.stdout.write(text)
    sys.stdout.flush()


def get_layer_by_param_name(model, param_name): # model로 부터 측정 레이어를 얻음 layer parameter name으로 부터
    '''
        Get a certain layer (e.g. torch.Conv2d) from a model
        by layer parameter name (e.g. models.conv_layers.0.weight)
        
        Input: 
            `model`: model we want to get a certain layer from
            `param_name`: (string) layer parameter name
            
        Output: 
            `layer`: (e.g. torch.nn.Conv2d)
    '''
    # Get layer from model using layer name.
    layer_name_str_split = param_name.split(STRING_SEPARATOR)[:-1]
    layer = model
    for s in layer_name_str_split:
        layer = getattr(layer, s)
    return layer


def get_keys_from_ordered_dict(ordered_dict): # ordered dict으로부터  key의 orderd list를 얻음
    '''
        get ordered list of keys from ordered dict
        
        Input: 
            `ordered_dict`
            
        Output:
            `dict_keys`
    '''
    dict_keys = []
    for key, _ in ordered_dict.items():
        dict_keys.append(key)  # get key from (key, value) pair
    return dict_keys


def extract_feature_map_sizes(model, input_data_shape): # conv와 fc laterwise  feature map size를 얻음
    # get_network_def_from_model(model, input_data_shape)에서
    # feature map size를 extract할때 사용
    # fmap_sizes_dict = extract_feature_map_sizes(model, input_data_shape)
    '''
        get conv and fc layerwise feature map size
        
        Input:
            `model`: model which we want to get layerwise feature map size.
            `input_data_shape`: (list) [C, H, W].
        
        Output:
            `fmap_sizes_dict`: (dict) layerwise feature map sizes.
        
    '''
    fmap_sizes_dict = {}
    hooks = []
    model = model.cuda()
    model.eval()

    def _register_hook(module):
        def _hook(module, input, output):
            type_str = module.__class__.__name__
            if type_str in (CONV_LAYER_TYPES + FC_LAYER_TYPES):
                module_id = id(module)
                in_fmap_size = list(input[0].size())
                out_fmap_size = list(output.size())
                fmap_sizes_dict[module_id] = {KEY_INPUT_FEATURE_MAP_SIZE: in_fmap_size,
                                              KEY_OUTPUT_FEATURE_MAP_SIZE: out_fmap_size}

        if (not isinstance(module, torch.nn.Sequential) and not isinstance(module, torch.nn.ModuleList) and not (
                module == model)):
            hooks.append(module.register_forward_hook(_hook))

    model.apply(_register_hook)
    _ = model(torch.randn([1, *input_data_shape]).cuda())
    for hook in hooks:
        hook.remove()

    return fmap_sizes_dict


def get_network_def_from_model(model, input_data_shape): # input model의 network def(OrderedDict= 삽입된 순서 기억)을 리턴
    '''
        return network def (OrderedDict) of the input model
        
        network_def only contains information about FC, Conv2d, ConvTranspose2d
        not includes batchnorm ...
  
        Input: 
            `model`: model we want to get network_def from
            `input_data_shape`: (list) [C, H, W].
        
        Output:
            `network_def`: (OrderedDict)
                           keys(): layer name (e.g. model.0.1, feature.2 ...)
                           values(): layer properties (dict)
    '''
    network_def = OrderedDict()
    state_dict = model.state_dict()

    # extract model keys in ordered manner from model dict. key, value를 얻어 변수에 넣어줌
    state_dict_keys = get_keys_from_ordered_dict(state_dict) # 140 -> 150줄

    # extract the feature map sizes. 변수에 넣어줌
    fmap_sizes_dict = extract_feature_map_sizes(model, input_data_shape)
    
    # for pixel shuffle, 밑의 3개의 변수는 pixel shuffle과 관련된 변수
    previous_layer_name_str = None
    previous_out_channels = None
    before_squared_pixel_shuffle_factor = int(1)

    for layer_param_name in state_dict_keys:
        layer = get_layer_by_param_name(model, layer_param_name)
        layer_id = id(layer)
        layer_name_str = STRING_SEPARATOR.join(layer_param_name.split(STRING_SEPARATOR)[:-1])
        layer_type_str = layer.__class__.__name__

        # If conv layer, populate network definition.
        # WARNING: ignores maxpool and upsampling layers.
        if layer_type_str in (CONV_LAYER_TYPES + FC_LAYER_TYPES) and WEIGHTSTRING in layer_param_name:

            # Populate network def. network def를 채움 : fc_later_type에 해당될 때
            if layer_type_str in FC_LAYER_TYPES:
                # netwokr_def를 채움
                network_def[layer_name_str] = {
                    KEY_IS_DEPTHWISE: False,
                    KEY_NUM_IN_CHANNELS: layer.in_features,
                    KEY_NUM_OUT_CHANNELS: layer.out_features,
                    KEY_KERNEL_SIZE: (1, 1),
                    KEY_STRIDE: (1, 1),
                    KEY_PADDING: (0, 0),
                    KEY_GROUPS: 1,
                    KEY_INPUT_FEATURE_MAP_SIZE: [1, fmap_sizes_dict[layer_id][KEY_INPUT_FEATURE_MAP_SIZE][1], 1, 1],
                    KEY_OUTPUT_FEATURE_MAP_SIZE: [1, fmap_sizes_dict[layer_id][KEY_OUTPUT_FEATURE_MAP_SIZE][1], 1, 1]
                }
            else: # this means layer_type_str is in CONV_LAYER_TYPES # conv_layer_type에 해당될 때

                # Note: Need to handle the special case when there is only one filter in the depth-wise layer
                #       because the number of groups will also be 1, which is the same as that of the point-wise layer.
                if layer.groups == 1:
                    is_depthwise = False
                else:
                    is_depthwise = True
                # network_def를 채움
                network_def[layer_name_str] = {
                    KEY_IS_DEPTHWISE: is_depthwise,
                    KEY_NUM_IN_CHANNELS: layer.in_channels,
                    KEY_NUM_OUT_CHANNELS: layer.out_channels,
                    KEY_KERNEL_SIZE: layer.kernel_size,
                    KEY_STRIDE: layer.stride,
                    KEY_PADDING: layer.padding,
                    KEY_GROUPS: layer.groups,
                    
                    # (1, C, H, W)
                    KEY_INPUT_FEATURE_MAP_SIZE: fmap_sizes_dict[layer_id][KEY_INPUT_FEATURE_MAP_SIZE],
                    KEY_OUTPUT_FEATURE_MAP_SIZE: fmap_sizes_dict[layer_id][KEY_OUTPUT_FEATURE_MAP_SIZE]
                } # fmap_sizes_dict = extract_feature_map_sizes(model, input_data_shape), fmap_sizes_dict: (dict) layerwise feature map sizes
            network_def[layer_name_str][KEY_LAYER_TYPE_STR] = layer_type_str
            

    # Support pixel shuffle. pixel shuffle 지원
            if layer_type_str in FC_LAYER_TYPES: # fc_later_type에 해당될 때
                before_squared_pixel_shuffle_factor = int(1) # scale factor로 사용됨
            else:
                if previous_out_channels is None: # previous_out_channel이 none이면 처음 루프에 들어왔을 때(초기값이 none임)
                    before_squared_pixel_shuffle_factor = int(1)
                else:
                    if previous_out_channels % layer.in_channels != 0:
                        raise ValueError('previous_out_channels is not divisible by layer.in_channels.')
                    before_squared_pixel_shuffle_factor = int(previous_out_channels / layer.in_channels)
                    # 이전레이어의 출력을 현재 레이어의 입력으로 나눠줌  128 / 32 = 4
                previous_out_channels = layer.out_channels # 현재 채널의 출력 채널 개수로 이전 채널의 출력 개수를 업데이트 시켜줌
            if previous_layer_name_str is not None:
                network_def[previous_layer_name_str][
                    KEY_AFTER_SQUSRED_PIXEL_SHUFFLE_FACTOR] = before_squared_pixel_shuffle_factor
            network_def[layer_name_str][KEY_BEFORE_SQUARED_PIXEL_SHUFFLE_FACTOR] = before_squared_pixel_shuffle_factor
            previous_layer_name_str = layer_name_str

    if previous_layer_name_str:
        network_def[previous_layer_name_str][
        KEY_AFTER_SQUSRED_PIXEL_SHUFFLE_FACTOR] = before_squared_pixel_shuffle_factor

    return network_def # network_def 리턴,  network_def[layer_name_str][KEY_LAYER_TYPE_STR]이런 형태


def compute_weights_and_macs(network_def): # 전체 네트워크의 wieght, mac의 값을 계산
    '''
        Compute the number of weights and MACs of a whole network.
        
        Input: 
            `network_def`: defined in get_network_def_from_model()
        
        Output:
            `layer_weights_dict`: (OrderedDict) records layerwise num of weights.
            `total_num_weights`: (int) total num of weights. 
            `layer_macs_dict`: (OrderedDict) recordes layerwise num of MACs.
            `total_num_macs`: (int) total num of MACs.     
    '''
    total_num_weights, total_num_macs = 0, 0 # 변수 초기화

    # Extract conv layer names from ordered network dict.
    network_def_keys = get_keys_from_ordered_dict(network_def)

    # Init dict to store num resources for each layer.
    layer_weights_dict = OrderedDict()
    layer_macs_dict = OrderedDict()

    # Iterate over conv layers in network def.
    for layer_name in network_def_keys:
        # Take product of filter size dimensions to get num weights for layer.
        layer_num_weights = (network_def[layer_name][KEY_NUM_OUT_CHANNELS] / \
                             network_def[layer_name][KEY_GROUPS]) * \
                            network_def[layer_name][KEY_NUM_IN_CHANNELS] * \
                            network_def[layer_name][KEY_KERNEL_SIZE][0] * \
                            network_def[layer_name][KEY_KERNEL_SIZE][1]

        # Store num weights in layer dict and add to total. weight를 계산하는 부분
        layer_weights_dict[layer_name] = layer_num_weights
        total_num_weights += layer_num_weights
        
        # Determine num macs for layer using output size. mac을 계산하는 부분
        output_size = network_def[layer_name][KEY_OUTPUT_FEATURE_MAP_SIZE]
        output_height, output_width = output_size[2], output_size[3]
        layer_num_macs = layer_num_weights * output_width * output_height

        # Store num macs in layer dict and add to total.
        layer_macs_dict[layer_name] = layer_num_macs
        total_num_macs += layer_num_macs

    return layer_weights_dict, total_num_weights, layer_macs_dict, total_num_macs # 리턴


def measure_latency(model, input_data_shape, runtimes=500): # 모델의 latency 측정
    '''
        Measure latency of 'model'
        
        Randomly sample 'runtimes' inputs with normal distribution and
        measure the latencies
    
        Input: 
            `model`: model to be measured (e.g. torch.nn.Conv2d)
            `input_shape`: (list) input shape of the model (e.g. (B, C, H, W))
           
        Output: 
            average time (float)
    '''
    total_time = .0
    is_cuda = next(model.parameters()).is_cuda
    if is_cuda: 
        cuda_num = next(model.parameters()).get_device()
    for i in range(runtimes):       
        if is_cuda:
            input = torch.cuda.FloatTensor(*input_data_shape).normal_(0, 1)
            input = input.cuda(cuda_num)    
            with torch.no_grad():
                start = time.time()
                model(input)
                torch.cuda.synchronize()
                finish = time.time()
        else:
            input = torch.randn(input_data_shape)
            with torch.no_grad():
                start = time.time()
                model(input)
                finish = time.time()
        total_time += (finish - start)
    return total_time/float(runtimes)


def compute_latency_from_lookup_table(network_def, lookup_table_path): # network_def에서 정의된 모든 층에대한 latency 계산
    '''
        Compute the latency of all layers defined in `network_def` (only including Conv and FC).
        
        When the value of latency is not in the lookup table, that value would be interpolated.
        
        Input:
            `network_def`: defined in get_network_def_from_model()
            `lookup_table_path`: (string) path to lookup table
        
        Output: 
            `latency`: (float) latency
    '''
    latency = .0  # latency 초기화
    with open(lookup_table_path, 'rb') as file_id: # lookup_table 열기
        lookup_table = pickle.load(file_id)
    for layer_name, layer_properties in network_def.items():
        if layer_name not in lookup_table.keys(): # layer_name이 lookup_table의 key에 없으면
            raise ValueError('Layer name {} in network def not found in lookup table'.format(layer_name))
            break # 에러 출력
        num_in_channels = layer_properties[KEY_NUM_IN_CHANNELS] # num_in_channels 초기화
        num_out_channels = layer_properties[KEY_NUM_OUT_CHANNELS] # num_out_channels 초기화
        if (num_in_channels, num_out_channels) in lookup_table[layer_name][KEY_LATENCY].keys():
            latency += lookup_table[layer_name][KEY_LATENCY][(num_in_channels, num_out_channels)] # lookup_table에 있으면 latency 증가
        else:
            # Not found in the lookup table, then interpolate the latency
            # num_in_channel, num_out_channels이 lookup_table에 없으면 interpolate(실험적이나 통계적으로 구해진 것을 이용해 값추정)
            feature_samples = np.array(list(lookup_table[layer_name][KEY_LATENCY].keys())) # feature_samples 초기화
            feature_samples_in = feature_samples[:, 0] # 초기화된 feature_samples들 중 앞에것들에 대한 모든정도를 feature_samples_in으로
            feature_samples_out = feature_samples[:, 1] # 초기화된 feature_samples들 중 뒤에것들에 대한 모든정도를 feature_samples_out으로
            measurement = np.array(list(lookup_table[layer_name][KEY_LATENCY].values())) # measurement 초기화
            assert feature_samples_in.shape == feature_samples_out.shape # shape이 같지 않으면 error 발생
            assert feature_samples_in.shape == measurement.shape # shape이 같지 않으면 error 발생
            rbf = Rbf(feature_samples_in, feature_samples_out, \
                      measurement, function='cubic') #
            num_in_channels = np.array([num_in_channels])
            num_out_channels = np.array([num_out_channels])
            estimated_latency = rbf(num_in_channels, num_out_channels) # in, out 채널로 rbf를 통해 latency 측정
            latency += estimated_latency[0] # 구한 latency값을 더해줌
    return latency # latency 리턴


def compute_resource(network_def, resource_type, lookup_table_path=None): # resource type에 대해 resoucre 계산
    '''
        compute resource based on resource type
        
        Input:
            `network_def`: defined in get_network_def_from_model()
            `resource_type`: (string) (FLOPS/WEIGHTS/LATENCY)
            `lookup_table_path`: (string) path to lookup table
        
        Output:
            `resource`: (float)
    '''
    # flops, weights, latency에대해 network_def로 부터 계산, latency일때만 lookup_table_path 사용
    if resource_type == 'FLOPS':
        _, _, _, resource = compute_weights_and_macs(network_def) # latency 계산
    elif resource_type == 'WEIGHTS':
        _, resource, _, _ = compute_weights_and_macs(network_def) # latency 계산
    elif resource_type == 'LATENCY':
        resource = compute_latency_from_lookup_table(network_def, lookup_table_path) # latency 계산
    else:
        raise ValueError('Only support the resource type `FLOPS`, `WEIGHTS`, and `LATENCY`.') # 3가지 타입이 아닐시 오류
    return resource # resource 리턴


def build_latency_lookup_table(network_def_full, lookup_table_path, min_conv_feature_size=8, 
                       min_fc_feature_size=128, measure_latency_batch_size=4, 
                       measure_latency_sample_times=500, verbose=False): #  층들의 latency에 대한 lookup table을 만든다
    '''
        Build lookup table for latencies of layers defined by `network_def_full`.
        
        Supported layers: Conv2d, Linear, ConvTranspose2d
            
        Modify get_network_def_from_model() and this function to include more layer types.
            
        input: 
            `network_def_full`: defined in get_network_def_from_model()
            `lookup_table_path`: (string) path to save the file of lookup table
            `min_conv_feature_size`: (int) The size of feature maps of simplified layers (conv layer)
                along channel dimmension are multiples of 'min_conv_feature_size'.
                The reason is that on mobile devices, the computation of (B, 7, H, W) tensors 
                would take longer time than that of (B, 8, H, W) tensors.
            `min_fc_feature_size`: (int) The size of features of simplified FC layers are 
                multiples of 'min_fc_feature_size'.
            `measure_latency_batch_size`: (int) the batch size of input data
                when running forward functions to measure latency.
            `measure_latency_sample_times`: (int) the number of times to run the forward function of 
                a layer in order to get its latency.
            `verbose`: (bool) set True to display detailed information.
    '''
    
    resource_type = 'LATENCY' # resource type이 latenct일때 사용
    # Generate the lookup table.
    lookup_table = OrderedDict()
    for layer_name, layer_properties in network_def_full.items():
        
        if verbose: # 자세한 로깅을 출력할지 말지
            print('-------------------------------------------')
            print('Measuring layer', layer_name, ':')
        
        # If the layer has the same properties as a previous layer, directly use the previous lookup table.
        for layer_name_pre, layer_properties_pre in network_def_full.items(): # layer_name : 이름 값, layer_properites : 속성값
            if layer_name_pre == layer_name: # 이전 레이어의 이름과 이름이 같으면
                break

            # Do not consider pixel shuffling.
            layer_properties_pre[KEY_BEFORE_SQUARED_PIXEL_SHUFFLE_FACTOR] = layer_properties[
                KEY_BEFORE_SQUARED_PIXEL_SHUFFLE_FACTOR]
            layer_properties_pre[KEY_AFTER_SQUSRED_PIXEL_SHUFFLE_FACTOR] = layer_properties[
                KEY_AFTER_SQUSRED_PIXEL_SHUFFLE_FACTOR]

            if layer_properties_pre == layer_properties:
                lookup_table[layer_name] = lookup_table[layer_name_pre]
                if verbose:
                    print('    Find previous layer', layer_name_pre, 'that has the same properties')
                break
        if layer_name in lookup_table:
            continue

        is_depthwise = layer_properties[KEY_IS_DEPTHWISE]
        num_in_channels = layer_properties[KEY_NUM_IN_CHANNELS]
        num_out_channels = layer_properties[KEY_NUM_OUT_CHANNELS]
        kernel_size = layer_properties[KEY_KERNEL_SIZE]
        stride = layer_properties[KEY_STRIDE]
        padding = layer_properties[KEY_PADDING]
        groups = layer_properties[KEY_GROUPS]
        layer_type_str = layer_properties[KEY_LAYER_TYPE_STR]
        input_data_shape = layer_properties[KEY_INPUT_FEATURE_MAP_SIZE]
        
        
        lookup_table[layer_name] = {}
        lookup_table[layer_name][KEY_IS_DEPTHWISE]      = is_depthwise
        lookup_table[layer_name][KEY_NUM_IN_CHANNELS]   = num_in_channels
        lookup_table[layer_name][KEY_NUM_OUT_CHANNELS]  = num_out_channels
        lookup_table[layer_name][KEY_KERNEL_SIZE]       = kernel_size
        lookup_table[layer_name][KEY_STRIDE]            = stride
        lookup_table[layer_name][KEY_PADDING]           = padding
        lookup_table[layer_name][KEY_GROUPS]            = groups
        lookup_table[layer_name][KEY_LAYER_TYPE_STR]    = layer_type_str
        lookup_table[layer_name][KEY_INPUT_FEATURE_MAP_SIZE] = input_data_shape
        lookup_table[layer_name][KEY_LATENCY]           = {}
        
        print('Is depthwise:', is_depthwise)
        print('Num in channels:', num_in_channels)
        print('Num out channels:', num_out_channels)
        print('Kernel size:', kernel_size)
        print('Stride:', stride)
        print('Padding:', padding)
        print('Groups:', groups)
        print('Input feature map size:', input_data_shape)
        print('Layer type:', layer_type_str)
        
        '''
        if num_in_channels >= min_feature_size and \
            (num_in_channels % min_feature_size != 0 or num_out_channels % min_feature_size != 0):
            raise ValueError('The number of channels is not divisible by {}.'.format(str(min_feature_size)))
        '''
        
        if layer_type_str in CONV_LAYER_TYPES:   # layer type이 conv일 때
            min_feature_size = min_conv_feature_size    # min_feature size초기화(이 변수는 여기서 처음 등장함), 오른쪽 변수는 하이퍼 마라미터로 피처맵을 축소하는 최소크기 결정함, 값이 너무 작아져 정보가 손실되지 않도록
        elif layer_type_str in FC_LAYER_TYPES:    # layer type이 fc일 때
            min_feature_size = min_fc_feature_size
        else: # type에 해당되지 않으면 에러 출력
            raise ValueError('Layer type {} not supported'.format(layer_type_str))
        
        for reduced_num_in_channels in range(num_in_channels, 0, -min_feature_size): # reduced_num_in_channel 처음 등장, min_feature_size만큼 줄여감
            if verbose: # 세부사항 출력과 관련된 부분
                index = 1
                print('    Start measuring num_in_channels =', reduced_num_in_channels)
            
            if is_depthwise: # depthwise이면 reduced_num_out_channels_list값을 reduced_num_in_channels 값인 단일 요소를 포함하는 리스트
                reduced_num_out_channels_list = [reduced_num_in_channels]
            else: # 아니면 num_out_channels에서 1까지의 일련의 숫자를 생성하며 스텝 크기는 -min_feature_size 즉, 출력 채널 수가 1에 도달할 때까지 각 단계에서 출력 채널 수가 min_feature_size만큼 감소
                reduced_num_out_channels_list = list(range(num_out_channels, 0, -min_feature_size))
                
            for reduced_num_out_channels in reduced_num_out_channels_list:
                if resource_type == 'LATENCY': # resource type이 latency이면
                    if layer_type_str == 'Conv2d': # 레이어 타입이 conv2d이면
                        if is_depthwise: # 레이어가 depthwise이면
                            layer_test = torch.nn.Conv2d(reduced_num_in_channels, reduced_num_out_channels, \
                            kernel_size, stride, padding, groups=reduced_num_in_channels)
                            # 2D conv로 새 인스턴스를 만듦
                        else:
                            layer_test = torch.nn.Conv2d(reduced_num_in_channels, reduced_num_out_channels, \
                            kernel_size, stride, padding, groups=groups) # 2D conv로 새 인스턴스를 만듦
                        input_data_shape = layer_properties[KEY_INPUT_FEATURE_MAP_SIZE]
                        # KEY_INPUT_FEATURE_MAP_SIZE 키를 사용하여 레이어의 입력 피쳐 맵 크기를 layer_properties dictionary에서 검색
                        input_data_shape = (measure_latency_batch_size, 
                            reduced_num_in_channels, *input_data_shape[2::])
                        # 입력 피쳐 맵의 공간 차원이 지정
                        # * 가 무엇을 의미하는가?
                    elif layer_type_str == 'Linear': # 레이어 타입이 linear이면
                        layer_test = torch.nn.Linear(reduced_num_in_channels, reduced_num_out_channels)
                        input_data_shape = (measure_latency_batch_size, reduced_num_in_channels)
                    elif layer_type_str == 'ConvTranspose2d': # 레이어 타입이 convtranspose2d이면
                        if is_depthwise:
                            layer_test = torch.nn.ConvTranspose2d(reduced_num_in_channels, reduced_num_out_channels, 
                                kernel_size, stride, padding, groups=reduced_num_in_channels)
                        else:
                            layer_test = torch.nn.ConvTranspose2d(reduced_num_in_channels, reduced_num_out_channels, 
                                kernel_size, stride, padding, groups=groups)
                        input_data_shape = layer_properties[KEY_INPUT_FEATURE_MAP_SIZE]
                        input_data_shape = (measure_latency_batch_size, 
                            reduced_num_in_channels, *input_data_shape[2::])
                    else:
                        raise ValueError('Not support this type of layer.')
                    if torch.cuda.is_available():
                        layer_test = layer_test.cuda()
                    measurement = measure_latency(layer_test, input_data_shape, measure_latency_sample_times)
                else:
                    raise ValueError('Only support building the lookup table for `LATENCY`.')


                # Add the measurement into the lookup table.
                lookup_table[layer_name][KEY_LATENCY][(reduced_num_in_channels, reduced_num_out_channels)] = measurement
                
                if verbose:
                    update_progress(index, len(reduced_num_out_channels_list), latency=str(measurement))
                    index = index + 1
                    
            if verbose:
                print(' ')
                print('    Finish measuring num_in_channels =', reduced_num_in_channels)
    # Save the lookup table.
    with open(lookup_table_path, 'wb') as file_id:
        pickle.dump(lookup_table, file_id)      
    return 


def simplify_network_def_based_on_constraint(network_def, block, constraint, resource_type,
                                             lookup_table_path=None, skip_connection_block_sets=[], 
                                             min_feature_size=8):
    '''
        Derive how much a certain block of layers ('block') should be simplified 
        based on resource constraints.
            
        Here we treat one block as one layer although a block can contain several layers.
            
        Input:
            `network_def`: simplifiable network definition (conv & fc). defined in self.get_network_def_from_model(...)
            `block`: (int) index of block to simplify
            `constraint`: (float) representing the FLOPs/weights/latency constraint the simplied model should satisfy
            `resource_type`: (string) `FLOPS`, `WEIGHTS`, or `LATENCY`
            `lookup_table_path`: (string) path to latency lookup table. Needed only when resource_type == 'LATENCY'
            `skip_connection_block_sets`: (list or tuple) the list of sets of blocks. Blocks in the same sets will have the 
                same number of output channels as the corresponding feature maps will be summed later. 
                (default: [])py -3.7 -m pip install torch
                For example, if the outputs of block 0 and block 4 are summed and 
                the outputs of block 1 and block 5 are summed, then
                skip_connection_block_sets = [(0, 4), (1, 5)] or ((0, 4), (1, 5)). 직접 설정이 가능
                Note that we currently support addition.
                
            `min_feature_size`: (int) the number of output channels of simplified (pruned) layer would be 
                multiples of min_feature_size. (default: 8)
        Output:
            `simplified_network_def`: simplified network definition. Indicates how much the network should
                be simplified/pruned.
            `simplified_resource`: (float) the estimated resource consumption of simplified models.
    '''
    # Check whether the block has a skip connection. block이 skip connection이 있는지 확인
    block = [block] # block = int(round(barLength*progress))값으로 updata progress쪽과 관련있음, round는 반올림
    for skip_connection_block_set in skip_connection_block_sets: # sets : 집합
        if block[0] in skip_connection_block_set:
            block = list(skip_connection_block_set)
            block.sort()
            break # 이 코드 블록은 현재 블록이 더 큰 스킵 연결 세트의 일부인지 여부를 식별하고 만약 있다면 전체 세트로 대체하여 결과 네트워크 구조가 스킵 연결 세트의 모든 블록을 포함하도록 합니다.
    print('    simplify_def> constraint: ', constraint)
    print('    simplify_def> target block:', block)

    # Find the target layer and other layers whose output would later be addedF to that of target layer
    # (i.e. skip connection) # skip connection 같은 target layer를 찾는것
    # (contains layer index) 
    target_layer_indices = [] # 추후에 수정 되상이 될 레이어 인덱스를 저장하기 위한 목록
    max_num_out_channels = None # 모든 컨볼루션 레이어의 최대 채널 출력 수
    block_counter = 0 #
    for layer_idx, (layer_name, layer_properties) in enumerate(network_def.items()): # 각 컨볼루션 레이어 마다 반복
        # Neglect the depthwise layers.
        if layer_properties[KEY_IS_DEPTHWISE]: # depthwise layer이면 continue(무시하고 다시 반복문으로)
            continue
        if block_counter == block[0]: # depthwise layer가 아닌 layer에 대해  block counter가 skip_connection 의 첫번째 block요소와 동일한지
            target_layer_indices.append(layer_idx) # 현재 layer의 idx를 target layer의 indices에 append
            if max_num_out_channels is not None: # max_num_out_channels가 값이 있는지 확인
                if max_num_out_channels != layer_properties[KEY_NUM_OUT_CHANNELS]: # 현재레이어의 출력 채널 수가 최대 채널 출력수와 같지않으면 오류 출력
                    print('The blocks involved in this skip connection do not have compatible numbers of output '
                          'channels.')
                    sys.stdout.flush() # 버퍼를 플러쉬, 버퍼가 정상적으로 대기하기 전에 버퍼에있는 모든 것을 터미널에 기록
            max_num_out_channels = layer_properties[KEY_NUM_OUT_CHANNELS] # max_num_out_channels을 현재 출력 채널 수로 설정, 값을 맞춰주기위해서
            print('    simplify_def> target layer: {}, layer index: {}'.format(layer_name, layer_idx))
            del block[0] #skip_connection의 첫번째 block index제거
            if not block: # block이 비었으면, 즉 skip_connection의 모든 블록이 처리되면
                break # 중단
        block_counter += 1 # block_counter 1증가, 다음 layer로 이동

    # Check target_layer_idx.
    if target_layer_indices is None: # 수정할 레이어가 없으면 오류 출력
        raise ValueError('`Block` seems out of bound.')

    # Determine the number of filters and the resource consumption.
    simplified_network_def = copy.deepcopy(network_def) # 기존 네트워크를 복사
    simplified_resource = None # simplify된 resource를 저장하는 변수
    return_with_constraint_satisfied = False # 제약조건이 충족 된지 안된지에 대한 변수
    if max_num_out_channels >= min_feature_size: # skip_connection의 최대 출력 채널 개수가 min_feature_size보다 크면
        # Try numbers of channels that are multiples of '_MIN_FEATURE_SIZE'.
        num_out_channels_try = list(range(max_num_out_channels // min_feature_size * min_feature_size, 
                                          min_feature_size - 1, -min_feature_size)) # 출력 채널의 후보목록을 채널의 수를 min_feature_size의 배수로 만들고 할당
    else:
        num_out_channels_try = [max_num_out_channels] # 출력 채널의 후보 목록을 작으면 지금까지 발견된 최대 출력 채널 수만 할당
    # 이 출력 후보 목록은 다음 loop에서 제약 조건을 충족하고 최대로 리소스 절약을 하는 출력 채널을 찾는데 사용
    '''   
        Update # of output channels of target layers.
           
        Update # of input/output channels of all depthwise layers between target layers and 
        other subsequent non-depthwise layers (assuming # of groups == # of input channels)
            
        Update # of input channels of one non-depthwise layer following the target layers.
    '''
    for current_num_out_channels in num_out_channels_try:  # Only allow multiple of '_MIN_FEATURE_SIZE'. current_num_out_channels이 min_feature_size의 배수일때만
        for target_layer_index in target_layer_indices:
            update_num_out_channels = True # output 채널을 변경해야할지 말아야할지에 대한 변수
            current_num_out_channels_after_pixel_shuffle = current_num_out_channels
            for layer_idx, (layer_name, layer_properties) in enumerate(simplified_network_def.items()):
                if layer_idx < target_layer_index:
                    continue
                
                # for the block to be simplified (# of output channels is simplified) output channel simplified
                if update_num_out_channels: # output 채널을 업데이트 시켜야할 경우
                    if not layer_properties[KEY_IS_DEPTHWISE]:  # depthwise layer가 아닌 경우
                        layer_properties[KEY_NUM_OUT_CHANNELS] = current_num_out_channels  # output 채널 업데이트
                        update_num_out_channels = False  # output 채널을 변경 해야할지 말아야 할지에 대한 변수
                        
                        print('    simplify_def>     layer {}: num of output channel changed to {}'.format(layer_name, str(current_num_out_channels)))
                    else: # depthwise layer가 들어왔을 때
                        raise ValueError('Expected a non-depthwise layer but got a depthwise layer.')
                # for blocks following the target blocks (# of input channels is simplified) 입력채널 simplified
                else:
                    if current_num_out_channels_after_pixel_shuffle % layer_properties[
                        KEY_BEFORE_SQUARED_PIXEL_SHUFFLE_FACTOR] != 0:
                        raise ValueError('current_num_out_channels or current_num_out_channels_after_pixel_shuffle is '
                                         'not divisible by the scaling factor of pixel shuffling.')
                    current_num_out_channels_after_pixel_shuffle = (
                            current_num_out_channels_after_pixel_shuffle / layer_properties[
                        KEY_BEFORE_SQUARED_PIXEL_SHUFFLE_FACTOR])
                    layer_properties[KEY_NUM_IN_CHANNELS] = current_num_out_channels_after_pixel_shuffle
                    print('    simplify_def>     layer {}: num of input channel changed to {}'.format(layer_name, str(current_num_out_channels_after_pixel_shuffle)))

                    '''
                        Consider the case that a FC layer is placed after a Conv and Flatten:
                            FC: input feature size: Cin
                                output feature size: Cout
                            Conv: output feature map size: H x W x C
                            So Cin = H x W x C.
                            If C -> C' based on constraints, then Cin -> H x W x C'
                    '''
                    if layer_properties[KEY_BEFORE_SQUARED_PIXEL_SHUFFLE_FACTOR] == 1:
                        if network_def[layer_name][KEY_NUM_IN_CHANNELS] > max_num_out_channels:
                            assert network_def[layer_name][KEY_NUM_IN_CHANNELS] % max_num_out_channels == 0
                            # H x W here
                            spatial_factor = network_def[layer_name][KEY_NUM_IN_CHANNELS] // max_num_out_channels
                            layer_properties[KEY_NUM_IN_CHANNELS] = spatial_factor*current_num_out_channels
                            print('    simplify_def>     [Update] layer {}: num of input channel changed to {}'.format(layer_name, str(spatial_factor*current_num_out_channels)))

                    if not layer_properties[KEY_IS_DEPTHWISE]:
                        break
                    else:
                        layer_properties[KEY_NUM_OUT_CHANNELS] = current_num_out_channels_after_pixel_shuffle
                        layer_properties[KEY_GROUPS] = current_num_out_channels_after_pixel_shuffle
                        print('    simplify_def>     depthwise layer {}: num of output channel changed to {}'.format(layer_name, str(current_num_out_channels_after_pixel_shuffle)))


        # Get the current resource consumption
        simplified_resource = compute_resource(simplified_network_def, resource_type, 
                                               lookup_table_path) # float 값으로 현재 상태의 resource consumption값을 구함
        print('    simplify_def> finish trying num of output channel: {}, resource: {}'.format(current_num_out_channels, simplified_resource))
        
        # Terminate the simplification when the constraint has been satisfied.
        if simplified_resource < constraint: # constraint 값이 simplifed_source값보다 크면 중단
            return_with_constraint_satisfied = True
            print('    simplify_def> constraint {} met when trying num of output channel: {}'.format(constraint, current_num_out_channels))
            break

    if not return_with_constraint_satisfied: # constraint 값이 simplifed_source값보다 작을때
        warnings.warn(
            'Constraint not satisfied: constraint = {}, simplified_resource = {}'.format(constraint,
                                                                                         simplified_resource))
    return simplified_network_def, simplified_resource # 최종적으로 simplified_network_def, simplified_resource값 리턴


def simplify_model_based_on_network_def(simplified_network_def, model):
        '''
            Choose which filters to preserve
            
            Here filters with largest L2 magnitude will be kept
            
            Input:
                `simplified_network_def`: network_def shows how a model will be pruned.
                defined in get_network_def_from_model()
                
                `model`: model to be simplified.
                
            Output:
                `simplified_model`: simplified model.
        '''
        simplified_model = copy.deepcopy(model)
        simplified_state_dict = simplified_model.state_dict()
        kept_filter_idx = None

        for layer_param_full_name in simplified_state_dict.keys():
            layer = get_layer_by_param_name(simplified_model, layer_param_full_name) # layer_param_full_name에 의해 simplified_model로 부터 얻은 layer
            layer_param_full_name_split = layer_param_full_name.split(STRING_SEPARATOR) #  layer_param_full_name 문자열을 문자열 목록으로 나누기
            layer_name_str = STRING_SEPARATOR.join(layer_param_full_name_split[:-1]) # 현재 파라미터를 갖는 레이어 이름 생성
            layer_param_name = layer_param_full_name_split[-1] # 현재 파라미터 이름 생성
            layer_type_str = layer.__class__.__name__ # layer의 객체 class 이름을 가져옴

            # Reduce the number of input channels based on the layer and data type.
            # Reduce the number of biases of simplified layers
            if kept_filter_idx is None: # kept_filter_idx값이 none인지
                pass # 제거된 필터가 없음
            elif layer_type_str in CONV_LAYER_TYPES: # 레이어 타입이 conv이면
                # Support pixel shuffle.
                before_squared_pixel_shuffle_factor = simplified_network_def[layer_name_str][
                    KEY_BEFORE_SQUARED_PIXEL_SHUFFLE_FACTOR] #  simplified_network_def 딕셔너리에서 현재 레이어에 대한 픽셀 셔플 팩터를 검색
                kept_filter_idx = (kept_filter_idx[::before_squared_pixel_shuffle_factor] /
                                   before_squared_pixel_shuffle_factor)

                if layer_param_name == WEIGHTSTRING: # WEIGHTSTRING == layer_param_name:
                    if layer.groups == 1:  # Pointwise layer or depthwise layer with only one filter.
                        setattr(layer, layer_param_name,
                                torch.nn.Parameter(getattr(layer, layer_param_name)[:, kept_filter_idx, :, :]))
                        layer.in_channels = len(kept_filter_idx)
                        print('    simplify_model> simplify Conv layer {}: ipnut channel weights {}'.format(layer_name_str,
                          len(kept_filter_idx)))
                    else: # depthwise
                        setattr(layer, layer_param_name,
                                torch.nn.Parameter(getattr(layer, layer_param_name)[kept_filter_idx]))
                        layer.in_channels = len(kept_filter_idx)
                        layer.out_channels = len(kept_filter_idx)
                        layer.groups = len(kept_filter_idx)
                        print('    simplify_model> simplify Conv layer {}: ipnut/output channel weights {} and groups {}'.format(layer_name_str,
                          len(kept_filter_idx), len(kept_filter_idx)))
                elif layer_param_name == BIASSTRING: #BIASSTRING == layer_param_name:
                    print('    simplify_model> simplify Conv layer {}: output channel biases {}'.format(layer_name_str, 
                          len(kept_filter_idx)))
                    setattr(layer, layer_param_name,
                            torch.nn.Parameter(getattr(layer, layer_param_name)[kept_filter_idx]))
                else:
                    raise ValueError('The layer_param_name `{}` is not supported.'.format(layer_param_name))
            elif layer_type_str in FC_LAYER_TYPES:
                if layer_param_name == BIASSTRING:
                    print('    simplify_model> simplify FC layer {}: output channel biases {}'.format(layer_name_str,
                          len(kept_filter_idx)))
                    # the weights of this layer is already reduced
                    setattr(layer, layer_param_name, 
                            torch.nn.Parameter(getattr(layer, layer_param_name)[kept_filter_idx]))
                else:
                    '''
                        the input features should be modified 
                        as its previous layer has different output features
                    
                        Consider the case that a FC layer is placed after a Conv and Flatten:
                        FC: input feature size: Cin
                            output feature size: Cout
                        Conv: output feature map size: H x W x C
                        So Cin = H x W x C.
                        If C -> C' based on constraints, then Cin -> H x W x C'
                    '''
                    num_in_features = simplified_network_def[layer_name_str][KEY_NUM_IN_CHANNELS]
                   
                    if num_in_features > len(kept_filter_idx):
                        assert num_in_features % len(kept_filter_idx) == 0
                        # H x W here
                        spatial_ratio = int(num_in_features / len(kept_filter_idx))
                        kept_filter_idx_fc = kept_filter_idx.clone()
                        kept_filter_idx_fc_element = kept_filter_idx_fc*spatial_ratio
                        kept_filter_idx_fc = kept_filter_idx_fc_element.clone()
                        for i in range(1, spatial_ratio):
                            kept_filter_idx_fc = torch.cat((kept_filter_idx_fc, 
                                                            kept_filter_idx_fc_element + i), dim=0)
                        kept_filter_idx_fc, _ = kept_filter_idx_fc.sort()
                        setattr(layer, layer_param_name, 
                                torch.nn.Parameter(getattr(layer, layer_param_name)[:, kept_filter_idx_fc]))
                        layer.in_features = len(kept_filter_idx_fc)
                        assert len(kept_filter_idx_fc) == num_in_features

                    else:
                        setattr(layer, layer_param_name,
                            torch.nn.Parameter(getattr(layer, layer_param_name)[:, kept_filter_idx]))
                        layer.in_features = len(kept_filter_idx)
                        
                    print('    simplify_model> simplify FC layer {}: input channel weights {}'.format(layer_name_str,
                          layer.in_features))
                        
            elif layer_type_str in BNORM_LAYER_TYPES:
                if any(substr == layer_param_name for substr in [WEIGHTSTRING, BIASSTRING]):
                    setattr(layer, layer_param_name,
                            torch.nn.Parameter(getattr(layer, layer_param_name)[kept_filter_idx], requires_grad=True))
                    layer.num_features = len(kept_filter_idx)
                    print('    simplify_model> simplify {} layer {}: {} {}'.format(layer_type_str,
                          layer_name_str, layer_param_name, layer.num_features))
                elif any(substr == layer_param_name for substr in [RUNNING_MEANSTRING, RUNNING_VARSTRING]):
                    setattr(layer, layer_param_name,
                            torch.nn.Parameter(getattr(layer, layer_param_name)[kept_filter_idx], requires_grad=False))
                    layer.num_features = len(kept_filter_idx)
                    print('    simplify_model> simplify {} layer {}: {} {}'.format(layer_type_str,
                          layer_name_str, layer_param_name, layer.num_features))
                elif NUM_BATCHES_TRACKED == layer_param_name:
                    getattr(layer, layer_param_name).zero_()
                    print('    simplify_model> simplify {} layer {}: {} set to 0'.format(layer_type_str,
                          layer_name_str, layer_param_name))
                else:
                    raise ValueError('The layer_param_name `{}` is not supported.'.format(layer_param_name))
            else:
                raise ValueError('The layer type `{}` is not supported.'.format(type(layer)))

            # Reduce the number of filters and update kept_filter_idx if it is in network_def and
            # not a depth-wise layer.
            # Reduce the number of output feature maps of simplified layers
            if (layer_param_name == WEIGHTSTRING and
                    layer_name_str in simplified_network_def and
                    not simplified_network_def[layer_name_str][KEY_IS_DEPTHWISE]):
                num_filters = simplified_network_def[layer_name_str][KEY_NUM_OUT_CHANNELS]
                weight = layer.weight.data
                if num_filters == weight.shape[0]: 
                    # Not target layer thus not simplify
                    # Means the current model and simplified network def 
                    # have the same number of output channels
                    kept_filter_idx = None
                
                # Based on L2 norm, determine `kept_filter_idx`
                # `kept_filter_idx` is used to simplify the current layer (conv & fc) and
                # is also used to simplify some related following layers ()
                else:
                    after_squared_pixel_shuffle_factor = simplified_network_def[layer_name_str][
                        KEY_AFTER_SQUSRED_PIXEL_SHUFFLE_FACTOR]
                    if num_filters % after_squared_pixel_shuffle_factor != 0:
                        raise ValueError('num_filters is not divisible by after_squared_pixel_shuffle_factor.')
                    num_filters //= after_squared_pixel_shuffle_factor
                    
                    if layer_type_str in CONV_LAYER_TYPES:
                        filter_norm = (weight * weight).sum((1, 2, 3))
                        filter_norm = filter_norm.view(-1, after_squared_pixel_shuffle_factor).sum(1)
                    elif layer_type_str in FC_LAYER_TYPES:
                        filter_norm = (weight * weight).sum(1)
                    _, kept_filter_idx = filter_norm.topk(num_filters, sorted=False)
                    
                    # consider pixel shuffle
                    kept_filter_idx_element = kept_filter_idx * after_squared_pixel_shuffle_factor
                    kept_filter_idx = kept_filter_idx_element.clone()

                    for pixel_shuffle_factor_counter in range(1, after_squared_pixel_shuffle_factor):
                        kept_filter_idx = torch.cat(
                            (kept_filter_idx, kept_filter_idx_element + pixel_shuffle_factor_counter),
                            dim=0)
                    kept_filter_idx, _ = kept_filter_idx.sort()
                    
                    if layer_type_str in CONV_LAYER_TYPES:
                        setattr(layer, layer_param_name,
                            torch.nn.Parameter(getattr(layer, layer_param_name)[kept_filter_idx, :, :, :]))
                        layer.out_channels = len(kept_filter_idx)
                        
                        print('    simplify_model> simplify Conv layer {}: output channel weights {}'.format(layer_name_str,
                              len(kept_filter_idx)))
                    elif layer_type_str in FC_LAYER_TYPES:
                        setattr(layer, layer_param_name, 
                            torch.nn.Parameter(getattr(layer, layer_param_name)[kept_filter_idx, :]))  
                        layer.out_features = len(kept_filter_idx)
                        print('    simplify_model> simplify FC layer {}: output channel weights {}'.format(layer_name_str,
                              len(kept_filter_idx)))

        return simplified_model

