def color_info(path):
    from PIL import Image
    color_list = []
    for x in range(len(path)):
        color_list.append(Image.open(f'../real_vs_fake/real-vs-fake/{path[x]}').getbands())
    return color_list


def color_test(list):
    color_list = []
    for x in list:
        if x == ('R', 'G', 'B'):
            color_list.append(x)
    return color_list

def length_compare(var1, var2):
    if len(var1) == len(var2):
        return True
    else:
        return False