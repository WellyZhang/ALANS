# -*- code: utf-8 -*- 


import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

# settings
config = "center_single"
components = [0]

path = "/home/chizhang/Datasets/RAVEN-10000/"
files = glob.glob(os.path.join(path, config, "*.xml"))


def get_exist_number_rule():
    index_name = xml_rules[component_idx][0].attrib["name"]
    attrib_name = xml_rules[component_idx][0].attrib["attr"][:3]
    exist_rule_indicator = None
    exist_rule = None
    number_rule_indicator = None
    number_rule = None
    if index_name == "Constant":
        exist_rule_indicator = 1
        exist_rule = 0
        number_rule_indicator = 1
        number_rule = 0
    elif index_name == "Progression":
        if attrib_name == "Num":
            exist_rule_indicator = 0
            exist_rule = 0
            number_rule_indicator = 1
            number_rule = 0
        if attrib_name == "Pos":
            exist_rule_indicator = 1
            exist_rule = 0
            number_rule_indicator = 0
            number_rule = 0
    elif index_name == "Arithmetic":
        if attrib_name == "Num":
            exist_rule_indicator = 0
            exist_rule = 0
            number_rule_indicator = 1
            first = int(xml_panels[0][0][component_idx][0].attrib["Number"])
            second = int(xml_panels[1][0][component_idx][0].attrib["Number"])
            third = int(xml_panels[2][0][component_idx][0].attrib["Number"])
            if third == first + second + 1:
                number_rule = 1
            if third == first - second - 1:
                number_rule = 2
        if attrib_name == "Pos":
            exist_rule_indicator = 1
            exist_rule = 1
            number_rule_indicator = 0
            number_rule = 0
    elif index_name == "Distribute_Three":
        if attrib_name == "Num":
            exist_rule_indicator = 0
            exist_rule = 0
            number_rule_indicator = 1
            first = int(xml_panels[0][0][component_idx][0].attrib["Number"])
            second_left = int(xml_panels[5][0][component_idx][0].attrib["Number"])
            second_right = int(xml_panels[4][0][component_idx][0].attrib["Number"])
            if second_left == first:
                number_rule = 3
            if second_right == first:
                number_rule = 4
        if attrib_name == "Pos":
            exist_rule_indicator = 1
            number_rule_indicator = 0
            number_rule = 0
            all_position = eval(xml_panels[0][0][component_idx][0].attrib["Position"])
            first = []
            for entity in xml_panels[0][0][component_idx][0]:
                first.append(all_position.index(eval(entity.attrib["bbox"])))
            second_left = []
            for entity in xml_panels[5][0][component_idx][0]:
                second_left.append(all_position.index(eval(entity.attrib["bbox"])))
            second_right = []
            for entity in xml_panels[4][0][component_idx][0]:
                second_right.append(all_position.index(eval(entity.attrib["bbox"])))
            if set(second_left) == set(first):
                exist_rule = 2
            if set(second_right) == set(first):
                exist_rule = 3
    else:
        raise ValueError(file)
    assert exist_rule_indicator is not None, file
    assert number_rule_indicator is not None, file
    assert exist_rule is not None, file
    assert number_rule is not None, file
    return exist_rule_indicator, number_rule_indicator, exist_rule, number_rule

def get_type_rule():
    index_name = xml_rules[component_idx][1].attrib["name"]
    type_rule = None
    if index_name == "Constant":
        all_values = []
        for i in range(8):
            layout_values = []
            for j in range(len(xml_panels[i][0][component_idx][0])):
                layout_values.append(xml_panels[i][0][component_idx][0][j].attrib["Type"])
            if len(np.unique(layout_values)) > 1:
                all_values.append(5)
            else:
                all_values.append(layout_values[0])
        if 5 in all_values:
            type_rule = 5
        else:
            type_rule = 0
    elif index_name == "Progression":
        type_rule = 0
    elif index_name == "Arithmetic":
        first = int(xml_panels[0][0][component_idx][0][0].attrib["Type"])
        second = int(xml_panels[1][0][component_idx][0][0].attrib["Type"])
        third = int(xml_panels[2][0][component_idx][0][0].attrib["Type"])
        if third == first + second + 2:
            type_rule = 1
        if third == first - second - 2:
            type_rule = 2
    elif index_name == "Distribute_Three":
        first = int(xml_panels[0][0][component_idx][0][0].attrib["Type"])
        second_left = int(xml_panels[5][0][component_idx][0][0].attrib["Type"])
        second_right = int(xml_panels[4][0][component_idx][0][0].attrib["Type"])
        if second_left == first:
            type_rule = 3
        if second_right == first:
            type_rule = 4
    else:
        raise ValueError(file)
    assert type_rule is not None, file
    return type_rule

def get_size_rule():
    index_name = xml_rules[component_idx][2].attrib["name"]
    size_rule = None
    if index_name == "Constant":
        all_values = []
        for i in range(8):
            layout_values = []
            for j in range(len(xml_panels[i][0][component_idx][0])):
                layout_values.append(xml_panels[i][0][component_idx][0][j].attrib["Size"])
            if len(np.unique(layout_values)) > 1:
                all_values.append(6)
            else:
                all_values.append(layout_values[0])
        if 6 in all_values:
            size_rule = 5
        else:
            size_rule = 0
    elif index_name == "Progression":
        size_rule = 0
    elif index_name == "Arithmetic":
        first = int(xml_panels[0][0][component_idx][0][0].attrib["Size"])
        second = int(xml_panels[1][0][component_idx][0][0].attrib["Size"])
        third = int(xml_panels[2][0][component_idx][0][0].attrib["Size"])
        if third == first + second + 1:
            size_rule = 1
        if third == first - second - 1:
            size_rule = 2
    elif index_name == "Distribute_Three":
        first = int(xml_panels[0][0][component_idx][0][0].attrib["Size"])
        second_left = int(xml_panels[5][0][component_idx][0][0].attrib["Size"])
        second_right = int(xml_panels[4][0][component_idx][0][0].attrib["Size"])
        if second_left == first:
            size_rule = 3
        if second_right == first:
            size_rule = 4
    else:
        raise ValueError(file)
    assert size_rule is not None, file
    return size_rule

def get_color_rule():
    index_name = xml_rules[component_idx][3].attrib["name"]
    color_rule = None
    if index_name == "Constant":
        all_values = []
        for i in range(8):
            layout_values = []
            for j in range(len(xml_panels[i][0][component_idx][0])):
                layout_values.append(xml_panels[i][0][component_idx][0][j].attrib["Color"])
            if len(np.unique(layout_values)) > 1:
                all_values.append(10)
            else:
                all_values.append(layout_values[0])
        if 10 in all_values:
            color_rule = 5
        else:
            color_rule = 0
    elif index_name == "Progression":
        color_rule = 0
    elif index_name == "Arithmetic":
        first = int(xml_panels[0][0][component_idx][0][0].attrib["Color"])
        second = int(xml_panels[1][0][component_idx][0][0].attrib["Color"])
        third = int(xml_panels[2][0][component_idx][0][0].attrib["Color"])
        fourth = int(xml_panels[3][0][component_idx][0][0].attrib["Color"])
        fifth = int(xml_panels[4][0][component_idx][0][0].attrib["Color"])
        sixth = int(xml_panels[5][0][component_idx][0][0].attrib["Color"])
        if (third == first + second) and (sixth == fourth + fifth):
            color_rule = 1
        if (third == first - second) and (sixth == fourth - fifth):
            color_rule = 2
    elif index_name == "Distribute_Three":
        first = int(xml_panels[0][0][component_idx][0][0].attrib["Color"])
        second_left = int(xml_panels[5][0][component_idx][0][0].attrib["Color"])
        second_right = int(xml_panels[4][0][component_idx][0][0].attrib["Color"])
        if second_left == first:
            color_rule = 3
        if second_right == first:
            color_rule = 4
    else:
        raise ValueError(file)
    assert color_rule is not None, file
    return color_rule

for file in files:
    xml_tree = ET.parse(file)
    xml_tree_root = xml_tree.getroot()
    xml_panels = xml_tree_root[0]
    xml_rules = xml_tree_root[1]
    for component_idx in components:
        new_file = file.replace(".xml", "_comp{}_rule.npz".format(component_idx))
        exist_rule_indicator, number_rule_indicator, exist_rule, number_rule = get_exist_number_rule()
        type_rule = get_type_rule()
        size_rule = get_size_rule()
        color_rule = get_color_rule()
        np.savez(new_file, exist_rule_indicator=exist_rule_indicator,
                           number_rule_indicator=number_rule_indicator,
                           exist_rule=exist_rule,
                           number_rule=number_rule,
                           type_rule=type_rule,
                           size_rule=size_rule,
                           color_rule=color_rule)
