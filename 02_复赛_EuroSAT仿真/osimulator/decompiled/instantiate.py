Unsupported opcode: LOAD_ASSERTION_ERROR (101)
# Source Generated with Decompyle++
# File: instantiate.cpython-39.pyc (Python 3.9)

import argparse
import osimulator
import entrance
import shutil
import os
import torch
import numpy as np
import io

def random_test(model_instance):
    '''
    对给定的模型实例进行随机测试，计算输入向量与权重矩阵的乘积累加结果，并计算相应的统计指标。

    Args:
        model_instance: 模型实例对象，用于调用其matmul_omac_3d方法进行矩阵乘法运算。

    Returns:
        tuple: 包含四个元素的元组，依次为uint4图片精度统计指标、int4图片精度统计指标、uint4端到端精度统计指标、int4端到端精度统计指标。

    '''
    scaling_factor = np.zeros([
        1,
        model_instance.n_dimension])
    scaling_factor[(0, :)] = model_instance.scaling_factor
    print(f'''scaling_factor {scaling_factor}''')
    if model_instance.k_dimension <= 8:
        input_tensor = np.load(entrance.dir_cur + '/osimulator/datasets/input_vector_10000_8_uint4.npy').transpose((0, 2, 1))
    elif model_instance.k_dimension <= 16:
        input_tensor = np.load(entrance.dir_cur + '/osimulator/datasets/input_vector_10000_16_uint4.npy').transpose((0, 2, 1))
    elif model_instance.k_dimension <= 32:
        input_tensor = np.load(entrance.dir_cur + '/osimulator/datasets/input_vector_10000_32_uint4.npy').transpose((0, 2, 1))
    if model_instance.k_dimension <= 8:
        weight_tensor = np.load(entrance.dir_cur + '/osimulator/datasets/weight_matrix_10000_8_int4.npy')
    elif model_instance.k_dimension <= 16:
        weight_tensor = np.load(entrance.dir_cur + '/osimulator/datasets/weight_matrix_10000_16_int4.npy')
    elif model_instance.k_dimension <= 32:
        weight_tensor = np.load(entrance.dir_cur + '/osimulator/datasets/weight_matrix_10000_32_int4.npy')
    x_uint4 = torch.tensor(input_tensor.astype(np.int32))[(:, :, :model_instance.k_dimension)]
    print(f'''x_uint4 {x_uint4.shape}''')
    x_int4 = x_uint4 - 8
    x_uint2 = x_uint4 // 4
    x_int2 = x_int4 // 4
    y_int4 = torch.tensor(weight_tensor[(:, :model_instance.k_dimension, :model_instance.n_dimension)].astype(np.int32))
    y_int2 = y_int4 // 4
    if hasattr(model_instance, 'physical_k_dimension'):
        if model_instance.input_precision >= 4:
            (results, _) = model_instance(x_uint4, y_int4, 'uint4', None, **('x', 'y', 'inputType', 'seed'))
            expect = x_uint4 @ y_int4
            (uint4_pic_stats, uint4_end2end_stats) = entrance.stats(np.array(results), np.array(expect), scaling_factor, model_instance.input_precision, model_instance.output_precision, model_instance.physical_k_dimension)
            print('#########uint4##########')
            print(uint4_pic_stats)
            print(uint4_end2end_stats)
            (results, _) = model_instance(x_int4, y_int4, 'int4', None, **('x', 'y', 'inputType', 'seed'))
            expect = x_int4 @ y_int4
            (int4_pic_stats, int4_end2end_stats) = entrance.stats(np.array(results), np.array(expect), scaling_factor, model_instance.input_precision, model_instance.output_precision, model_instance.physical_k_dimension)
            print('#########int4##########')
            print(int4_pic_stats)
            print(int4_end2end_stats)
        else:
            (uint4_pic_stats, uint4_end2end_stats) = (None, None)
            (int4_pic_stats, int4_end2end_stats) = (None, None)
        if model_instance.input_precision >= 2:
            (results, _) = model_instance(x_uint2, y_int2, 'uint2', None, **('x', 'y', 'inputType', 'seed'))
            expect = x_uint2 @ y_int2
            (uint2_pic_stats, uint2_end2end_stats) = entrance.stats(np.array(results), np.array(expect), scaling_factor, model_instance.input_precision, model_instance.output_precision, model_instance.k_dimension)
            print('#########uint2##########')
            print(uint2_pic_stats)
            print(uint2_end2end_stats)
            (results, _) = model_instance(x_int2, y_int2, 'int2', None, **('x', 'y', 'inputType', 'seed'))
            expect = x_int2 @ y_int2
            (int2_pic_stats, int2_end2end_stats) = entrance.stats(np.array(results), np.array(expect), scaling_factor, model_instance.input_precision, model_instance.output_precision, model_instance.k_dimension)
            print('#########int2##########')
            print(int2_pic_stats)
            print(int2_end2end_stats)
        else:
            (uint2_pic_stats, uint2_end2end_stats) = (None, None)
            (int2_pic_stats, int2_end2end_stats) = (None, None)
    else:
        (results, _) = entrance.matmul_omac_3d(input_tensor.astype(np.int32), weight_tensor[(:, :model_instance.k_dimension, :model_instance.n_dimension)].astype(np.int32), 1, 1, None, model_instance, **('x', 'y', 'input_mapping_factor', 'wght_mapping_factor', 'seed', 'model_instance'))
        expect = input_tensor @ weight_tensor[(:, :model_instance.k_dimension, :model_instance.n_dimension)]
        (uint4_pic_stats, uint4_end2end_stats) = entrance.stats(np.array(results), np.array(expect), scaling_factor, model_instance.inbit, model_instance.outbit, model_instance.k_dimension)
        print('#########uint4##########')
        print(uint4_pic_stats)
        print(uint4_end2end_stats)
        (results, _) = entrance.matmul_omac_3d(input_tensor.astype(np.int32) - 8, weight_tensor[(:, :model_instance.k_dimension, :model_instance.n_dimension)].astype(np.int32), 1, 1, None, model_instance, **('x', 'y', 'input_mapping_factor', 'wght_mapping_factor', 'seed', 'model_instance'))
        expect = input_tensor - 8 @ weight_tensor[(:, :model_instance.k_dimension, :model_instance.n_dimension)]
        (int4_pic_stats, int4_end2end_stats) = entrance.stats(np.array(results), np.array(expect), scaling_factor, model_instance.inbit, model_instance.outbit, model_instance.k_dimension)
        print('#########int4##########')
        print(int4_pic_stats)
        print(int4_end2end_stats)
    return (uint4_pic_stats, int4_pic_stats, uint4_end2end_stats, int4_end2end_stats, uint2_pic_stats, int2_pic_stats, uint2_end2end_stats, int2_end2end_stats)


def int_or_none(value):
    if value.lower() == 'none':
        return None
    return None(value)


def set_4dac_vecmod(modulator_dac_enob = None, vec_mod = None):
    if modulator_dac_enob == 0:
        print(f'''vecmod.dac1.enob remain {vec_mod.dac1.enob}''')
        print(f'''vecmod.dac2.enob remain {vec_mod.dac2.enob}''')
        print(f'''vecmod.dac3.enob remain {vec_mod.dac3.enob}''')
        print(f'''vecmod.dac4.enob remain {vec_mod.dac4.enob}''')
    else:
        print(f'''before vecmod.dac1.enob {vec_mod.dac1.enob}''')
        print(f'''before vecmod.dac2.enob {vec_mod.dac3.enob}''')
        print(f'''before vecmod.dac3.enob {vec_mod.dac3.enob}''')
        print(f'''before vecmod.dac4.enob {vec_mod.dac4.enob}''')
        vec_mod.dac1.enob = modulator_dac_enob
        vec_mod.dac2.enob = modulator_dac_enob
        vec_mod.dac3.enob = modulator_dac_enob
        vec_mod.dac4.enob = modulator_dac_enob
        print(f'''after vecmod.dac1.enob {vec_mod.dac1.enob}''')
        print(f'''after vecmod.dac2.enob {vec_mod.dac2.enob}''')
        print(f'''after vecmod.dac3.enob {vec_mod.dac3.enob}''')
        print(f'''after vecmod.dac4.enob {vec_mod.dac4.enob}''')


def set_3dac_vecmod(modulator_dac_enob = None, vec_mod = None):
    if (modulator_dac_enob == 0 or vec_mod.dac1.enob == modulator_dac_enob) and vec_mod.dac3.enob == modulator_dac_enob and vec_mod.dac4.enob == modulator_dac_enob:
        print(f'''vecmod.dac1.enob remain {vec_mod.dac1.enob}''')
        print(f'''vecmod.dac3.enob remain {vec_mod.dac3.enob}''')
        print(f'''vecmod.dac4.enob remain {vec_mod.dac4.enob}''')
        return False
    None(f'''before vecmod.dac1.enob {vec_mod.dac1.enob}''')
    print(f'''before vecmod.dac3.enob {vec_mod.dac3.enob}''')
    print(f'''before vecmod.dac4.enob {vec_mod.dac4.enob}''')
    vec_mod.dac1.enob = modulator_dac_enob
    vec_mod.dac3.enob = modulator_dac_enob
    vec_mod.dac4.enob = modulator_dac_enob
    print(f'''after vecmod.dac1.enob {vec_mod.dac1.enob}''')
    print(f'''after vecmod.dac3.enob {vec_mod.dac3.enob}''')
    print(f'''after vecmod.dac4.enob {vec_mod.dac4.enob}''')
    return True


def set_2dac_weightmod(modulator_dac_enob = None, weight_mod = None):
    if (modulator_dac_enob == 0 or weight_mod.dac1.enob == modulator_dac_enob) and weight_mod.dac2.enob == modulator_dac_enob:
        print(f'''weightmod.dac1.enob remain {weight_mod.dac1.enob}''')
        print(f'''weightmod.dac2.enob remain {weight_mod.dac2.enob}''')
        return False
    None(f'''before weightmod.dac1.enob {weight_mod.dac1.enob}''')
    print(f'''before weightmod.dac2.enob {weight_mod.dac2.enob}''')
    weight_mod.dac1.enob = modulator_dac_enob
    weight_mod.dac2.enob = modulator_dac_enob
    print(f'''after weightmod.dac1.enob {weight_mod.dac1.enob}''')
    print(f'''after weightmod.dac2.enob {weight_mod.dac2.enob}''')
    return True


def set_acc_tia_noise(acc, acc_tia_noise):
    if acc_tia_noise == 0:
        print('acc.tia.noise remain {acc.tia.tia_noise_amp_div_sqrtclock}')
    else:
        print(f'''before acc.tia.noise {acc.tia.tia_noise_amp_div_sqrtclock}''')
        acc.tia.tia_noise_amp_div_sqrtclock = acc_tia_noise
        print(f'''after acc.tia.noise {acc.tia.tia_noise_amp_div_sqrtclock}''')


def set_acc_tia_gain(acc, acc_tia_gain, model_type, gain_scale):
    if model_type == 'pace2':
        acc_tia_gain = acc_tia_gain * gain_scale
    if acc_tia_gain == 0 or acc_tia_gain == acc.tia.gain:
        print('acc.tia.gain remain {acc.tia.gain}')
        return False
    None(f'''before acc.tia.gain {acc.tia.gain}''')
    acc.tia.gain = acc_tia_gain
    print(f'''after acc.tia.gain {acc.tia.gain}''')
    return True


def set_tx_tia_gain(tx, tx_gain):
    if tx_gain == 0 or tx_gain == tx.tia.gain:
        print(f'''tx.tia.gain remain {acc.tia.gain}''')
        return False
    None(f'''before tx.tia.gain {tx.tia.gain}''')
    tx.tia.gain = tx_gain
    print(f'''after acc.tia.gain {tx.tia.gain}''')
    return True


def set_acc_tia_mse(acc, clock_frequency, acc_tia_noise):
    if clock_frequency == 0:
        print('acc.tia.clock remain {acc.tia.system_clock}')
    else:
        print(f'''before acc.tia.clock {acc.tia.system_clock}''')
        acc.tia.system_clock = clock_frequency
        print(f'''after acc.tia.clock {acc.tia.system_clock}''')
    if acc_tia_noise == 0:
        print('acc.tia.noise remain {acc.tia.tia_noise_amp_div_sqrtclock}')
    else:
        print(f'''before acc.tia.noise {acc.tia.tia_noise_amp_div_sqrtclock}''')
        acc.tia.tia_noise_amp_div_sqrtclock = acc_tia_noise
        print(f'''after acc.tia.noise {acc.tia.tia_noise_amp_div_sqrtclock}''')
        print(f'''before acc.tia._noise_mse {acc.tia._noise_mse}''')
        acc.tia._noise_mse = acc.tia.tia_noise_amp_div_sqrtclock * entrance.sqrt(acc.tia.system_clock)
        print(f'''after acc.tia._noise_mse {acc.tia._noise_mse}''')


def set_acc_adc_fsr(acc, acc_adc_fsr):
    if acc_adc_fsr == 0 or acc_adc_fsr == acc.adc.fsr_volt:
        print(f'''acc.adc.fsr_volt remain {acc.adc.fsr_volt}''')
        return False
    None(f'''before acc.adc.fsr_volt {acc.adc.fsr_volt}''')
    acc.adc.fsr_volt = acc_adc_fsr
    print(f'''after acc.adc.fsr_volt {acc.adc.fsr_volt}''')
    return True


def set_laser_clock(laser, system_clock):
    _power_flunc = entrance.sqrt(10 ** (laser.rin / 10) * laser.system_clock)
    if system_clock == 0 or system_clock == laser.system_clock:
        print(f'''laser._power_flunc remain {_power_flunc}''')
        return False
    None(f'''before laser._power_flunc {_power_flunc}''')
    laser._power_flunc = entrance.sqrt(10 ** (laser.rin / 10) * laser.system_clock)
    print(f'''before laser._power_flunc {laser._power_flunc}''')
    return True


def set_pd_clock(weightmod, system_clock):
    if system_clock == 0 or system_clock == weightmod.wpd1.system_clock:
        print(f'''pd.system_clock remain {weightmod.wpd1.system_clock}''')
        return False
    None(f'''before pd.system_clock {weightmod.wpd1.system_clock}''')
    weightmod.wpd1.system_clock = system_clock
    weightmod.wpd2.system_clock = system_clock
    print(f'''before pd.system_clock {weightmod.wpd1.system_clock}''')
    return True


def create_args_table(args):
    '''
    创建并返回包含架构和组件属性的HTML表格字符串。
    
    Args:
        args (argparse.Namespace): 包含所需参数的命名空间对象，包括：
            - src_instance_path (str): 源实例路径，为空字符串时表示创建新实例。
            - dst_instance_path (str): 目标实例路径。
            - model_type (str): 模型类型。
            - laser_power (float): 激光功率。
            - k_dimension (int): k维度。
            - n_dimension (int): n维度。
            - input_precision (str): 输入精度。
            - weight_precision (str): 权重精度。
            - output_precision (str): 输出精度。
            - seed (int): 随机种子。
            - clock_frequency (float): TIA时钟频率。
            - acc_tia_gain (float): TIA增益。
            - acc_tia_noise (float): TIA噪声。
            - acc_adc_fsr (float): ADC满量程范围。
            - modulator_dac_enob (int): 调制器DAC的有效位数。
    
    Returns:
        str: 包含架构和组件属性的HTML表格字符串。
    
    '''
    arch_header = [
        'src_instance_path',
        'dst_instance_path',
        'model_type',
        'laser_power',
        'k_dimension',
        'n_dimension',
        'input_precision',
        'weight_precision',
        'output_precision',
        'seed']
    arch_attr = []
    arch_attr.append('new') if args.src_instance_path == '' else arch_attr.append(args.src_instance_path)
    arch_attr.append(args.dst_instance_path)
    arch_attr.append(args.model_type)
    arch_attr.append(args.laser_power)
    arch_attr.append(args.k_dimension)
    arch_attr.append(args.n_dimension)
    arch_attr.append(args.input_precision)
    arch_attr.append(args.weight_precision)
    arch_attr.append(args.output_precision)
    arch_attr.append(seed)
    component_header = [
        'clock_frequency',
        'acc_tia_gain',
        'acc_tia_noise',
        'acc_adc_fsr',
        'modulator_dac_enob']
    component_attr = []
    component_attr.append(args.clock_frequency)
    component_attr.append(args.acc_tia_gain)
    component_attr.append(args.acc_tia_noise)
    component_attr.append(args.acc_adc_fsr)
    component_attr.append(args.modulator_dac_enob)
    if os.path.exists(args.dst_instance_path + '/arch.csv'):
        os.remove(args.dst_instance_path + '/arch.csv')
    if os.path.exists(args.dst_instance_path + '/component.csv'):
        os.remove(args.dst_instance_path + '/component.csv')
    entrance.write_data_to_csv([
        arch_attr], args.dst_instance_path + '/arch.csv', arch_header)
    entrance.write_data_to_csv([
        component_attr], args.dst_instance_path + '/component.csv', component_header)
    arch_table = entrance.read_csv(args.dst_instance_path + '/arch.csv').transpose().to_html()
    component_table = entrance.read_csv(args.dst_instance_path + '/component.csv').transpose().to_html()
    tables = f'''\n    <p>Arch attributes</p>\n    {arch_table}\n    <br><br>\n    <p>Component attributes</p>\n    {component_table}\n    <br><br>\n    '''
    return tables


def create_random_test_table(pic_stats, end2end_stats, dst_instance_path, input_type):
    '''
    生成随机测试报告表格
    
    Args:
        pic_stats (dict): 图片统计信息字典，包含"pic enob", "pic std", "pic mean", "pic max", "pic min", "pic max_pos", "pic min_pos"等键值对
        end2end_stats (dict): 端到端统计信息字典，包含"enob", "std", "mean", "max", "min", "max_pos", "min_pos"等键值对
        dst_instance_path (str): 目标文件夹路径
        input_type (str): 输入类型，可选值为"uint4"或"int4"
    
    Returns:
        str: 随机测试报告表格的HTML格式字符串
    
    '''
    header = [
        'type',
        'enob',
        'std',
        'mean',
        'diff max',
        'diff min',
        'diff max pos',
        'diff min pos']
    attrs = [
        [
            'pic',
            pic_stats['pic enob'],
            pic_stats['pic std'],
            pic_stats['pic mean'],
            pic_stats['pic max'],
            pic_stats['pic min'],
            pic_stats['pic max_pos'],
            pic_stats['pic min_pos']],
        [
            'end2end',
            end2end_stats['enob'],
            end2end_stats['std'],
            end2end_stats['mean'],
            end2end_stats['max'],
            end2end_stats['min'],
            end2end_stats['max_pos'],
            end2end_stats['min_pos']]]
    if os.path.exists(dst_instance_path + '/random_test_report.csv'):
        os.remove(dst_instance_path + '/random_test_report.csv')
    entrance.write_data_to_csv(attrs, dst_instance_path + '/random_test_report.csv', header)
    random_test_report = entrance.read_csv(dst_instance_path + '/random_test_report.csv').to_html()
    if input_type == 'uint4':
        input_type = '[0, 15]'
        wght_type = '[-8, 7]'
    elif input_type == 'int4':
        input_type = '[-8, 7]'
        wght_type = '[-8, 7]'
    elif input_type == 'uint2':
        input_type = '[0, 3]'
        wght_type = '[-2, 1]'
    elif input_type == 'int2':
        input_type = '[-2, 1]'
        wght_type = '[-2, 1]'
    tables = f'''\n    <p>Random test report, dataset from solution team, input {input_type} random, weight {wght_type} random, statistics ENOB, STD/MEAN</p>\n    {random_test_report}\n    <br><br>\n    '''
    return tables


def create_lut_test_table(error, std, mean, out_bit, dst_instance_path, table_name):
    '''
    根据误差矩阵、标准差矩阵、均值矩阵、输出比特数、目标路径和表格名称创建LUT测试表格。
    
    Args:
        error (numpy.ndarray): 误差矩阵，shape为(b, m, n)
        std (numpy.ndarray): 标准差矩阵，shape为(n,)，其中n为通道数。
        mean (numpy.ndarray): 均值矩阵，shape为(n,)，其中n为通道数。
        out_bit (int): 输出比特数。
        dst_instance_path (str): 目标路径。
        table_name (str): 表格名称。
    
    Returns:
        str: 生成的表格字符串，包括表格名称、测试报告说明以及表格内容。
    
    '''
    header = [
        'channel',
        'enob',
        'std',
        'mean',
        'max',
        'min',
        'max pos',
        'min pos']
    attrs = []
    for n, std_n in enumerate(std):
        enob = np.log2(2 ** out_bit) - abs(np.log2(std_n))
        max = np.max(error[(:, :, n)])
        min = np.min(error[(:, :, n)])
        max_pos = error[(:, :, n)].argmax(0)
        min_pos = error[(:, :, n)].argmin(0)
        attrs.append([
            n,
            enob,
            std_n,
            mean[n],
            max,
            min,
            max_pos,
            min_pos])
    if os.path.exists(dst_instance_path + '/' + table_name):
        os.remove(dst_instance_path + '/' + table_name)
    entrance.write_data_to_csv(attrs, dst_instance_path + '/' + table_name, header)
    lut_test_report = entrance.read_csv(dst_instance_path + '/' + table_name).to_html()
    if table_name == 'wght_lut':
        tmp_str = 'Vector modulator input fixed 15, weight modulator input [-8, 7]'
    else:
        tmp_str = 'Vector modulator input [0, 15], weight modulator input fixed 7'
    tables = f'''\n    <p>{table_name} test report, {tmp_str}, statistics every channel\'s ENOB, STD/MEAN</p>\n    {lut_test_report}\n    '''
    return tables

# WARNING: Decompyle incomplete
