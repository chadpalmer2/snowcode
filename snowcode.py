from PIL import Image, ImageDraw, ImageEnhance, ImageStat, ImageFilter, ImageChops
import os
import sys
import numpy
from math import sqrt, sin, cos, atan2, pi
from reedsolo import RSCodec, ReedSolomonError
import cv2

import pdb

## hardcoded encoding sizes

encoding_size = 108
ecc_len = 27
data_size_limit = encoding_size - ecc_len

rsc = RSCodec(ecc_len)

## hardcoded coding features

spoke_count = 6
lines_per_spoke = 24
line_gradations = 8

### storage capacity in bytes = 2 * spoke_count * lines_per_spoke * log_2(line_gradations) / 8

## parameters to manipulate for image feature proportions

line_resolution = 20
spacing = 1

center_spoke_length = 12 * line_resolution
main_spoke_length = (lines_per_spoke * (1 + spacing) + 1) * line_resolution
endpoint_spoke_length = 6 * line_resolution

total_spoke_length = center_spoke_length + main_spoke_length + endpoint_spoke_length

image_size = int(total_spoke_length * 2.5)

## Helpful coordinate geometry functions

def get_translated_point(xy, d, angle):
    return (int(xy[0] + d * cos(angle)), int(xy[1] + d * sin(angle)))

def get_distance(xy1, xy2):
    x_side = xy2[0] - xy1[0]
    y_side = xy2[1] - xy1[1]

    return [
        sqrt(x_side**2 + y_side**2), 
        atan2(y_side, x_side)
    ]

# openCV function for finding hexagons - https://stackoverflow.com/questions/60177653/how-to-detect-an-octagonal-shape-in-python-and-opencv

def find_hexagons(image):
    # grayscale, threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    hex_centroids = []

    # find contours and detect valid hexagons
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for index, contour in enumerate(contours):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # check for hexagon
        if len(approx) != 6:
            continue

        # find centroid
        M = cv2.moments(contour)

        # eliminate corner cases/division by zero
        if M["m00"] == 0:
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        hex_centroids.append([(cX, cY), peri])

    # exit if insufficient number of hexagons found
    if len(hex_centroids) < 11:
        return []

    # get 11 largest hexagons
    hex_centroids.sort(reverse=True, key=lambda x: x[1])
    hex_centroids = hex_centroids[0:11]

    # get black_hexes
    middle = hex_centroids[0][0]
    black_hexes = []

    for h in hex_centroids[2:11]:
        x, y = h[0]
        if image[y][x][0] == 0:
            black_hexes.append(h[0])

    if len(black_hexes) != 6:
        return []

    black_hexes.sort()
    a, b, c = black_hexes[0], black_hexes[2], black_hexes[4]

    ## some silly math to figure out which black hex is which
    dist_ab = get_distance(a, b)[0]
    dist_bc = get_distance(b, c)[0]
    dist_ac = get_distance(a, c)[0]
    arr_min = min([dist_ab, dist_bc, dist_ac])
    arr_max = max([dist_ab, dist_bc, dist_ac])
    if arr_min == dist_ab:
        bottom = c
        if arr_max == dist_bc:
            top = b
            top_left = a
        else:
            top = a
            top_left = b
    elif arr_min == dist_bc:
        bottom = a
        if arr_max == dist_ab:
            top = b
            top_left = c
        else:
            top = c
            top_left = b
    else:
        bottom = b
        if arr_max == dist_bc:
            top = c
            top_left = a
        else:
            top = a
            top_left = c

    ## calculating approx location of bottom right hex and correcting from actual hexes
    trans_dist = get_distance(top_left, middle)
    bottom_right = get_translated_point(top_left, 2 * trans_dist[0], trans_dist[1])
    bottom_right = min(hex_centroids, key=lambda x: get_distance(bottom_right, x[0])[0])[0]

    # returning hex coordinates for perspective shift
    return [ top, top_left, bottom, bottom_right ]

# numpy helper function for perspective transform - https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = numpy.matrix(matrix, dtype=float)
    B = numpy.array(pb).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)

def perspective_transform(img, start_coords, end_coords):
    coeffs = find_coeffs(end_coords, start_coords)

    return img.transform((image_size, image_size), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)

# data encoding and decoding functions

def encode_payload(data):
    return rsc.encode(data)

def decode_payload(payload):
    return rsc.decode(payload)

def encode_line_lengths(payload):
    lengths = []
    parse_list = []
    for encoded_byte in payload:
        parse_list.append(encoded_byte)
        if len(parse_list) == 3:
            lengths += [
                parse_list[0] // 32,
                (parse_list[0] % 32) // 4,
                ((parse_list[0] % 4) * 2) + parse_list[1] // 128,
                (parse_list[1] % 128) // 16,
                (parse_list[1] % 16) // 2,
                ((parse_list[1] % 2) * 4) + parse_list[2] // 64,
                (parse_list[2] % 64) // 8,
                parse_list[2] % 8
            ]
            parse_list = []

    return lengths

def decode_line_lengths(lengths):
    payload = []
    parse_list = []
    for line in lengths:
        parse_list.append(line)
        if len(parse_list) == 8:
            payload += [
                (parse_list[0] * 32) + (parse_list[1] * 4) + (parse_list[2] // 2),
                ((parse_list[2] % 2) * 128) + (parse_list[3] * 16) + (parse_list[4] * 2) + (parse_list[5] // 4),
                ((parse_list[5] % 4) * 64) + (parse_list[6] * 8) + parse_list[7]
            ]
            parse_list = []

    return bytearray(payload)

# image building functions

def draw_hexagon(d, xy, spoke, color):
    d.polygon([
        get_translated_point(xy, spoke, pi / 6),
        get_translated_point(xy, spoke, pi / 2),
        get_translated_point(xy, spoke, 5 * pi / 6),
        get_translated_point(xy, spoke, 7 * pi / 6),
        get_translated_point(xy, spoke, 3 * pi / 2),
        get_translated_point(xy, spoke, 11 * pi / 6),
    ], fill=color)

def draw_full_hexagon(d, xy, spoke, fill_in=True):
    draw_hexagon(d, xy, spoke, (0, 0, 0))
    draw_hexagon(d, xy, spoke - line_resolution, (255, 255, 255))
    if fill_in:
        draw_hexagon(d, xy, spoke - 2 * line_resolution, (0, 0, 0))

def build_image(lengths):
    img = Image.new('RGB', (image_size, image_size), "white")
    d = ImageDraw.Draw(img)

    spoke_point_translation_angles = [
        pi / 2,
        3 * pi / 2,
        5 * pi / 6,
        pi / 6,
        7 * pi / 6,
        11 * pi / 6
    ]

    middle = (image_size // 2, image_size // 2)
    spoke_points = [ get_translated_point(middle, total_spoke_length, angle) for angle in spoke_point_translation_angles ]
    top, bottom, left_top, right_top, left_bottom, right_bottom = spoke_points

    d.line([top, bottom], fill=(0,0,0), width=line_resolution)
    d.line([left_top, right_bottom], fill=(0,0,0), width=line_resolution)
    d.line([right_top, left_bottom], fill=(0,0,0), width=line_resolution)

    draw_full_hexagon(d, middle, center_spoke_length)

    for point in [top, bottom, left_bottom]:
        draw_full_hexagon(d, point, endpoint_spoke_length, fill_in=True)

    for point in [left_top, right_top, right_bottom]:
        draw_full_hexagon(d, point, endpoint_spoke_length, fill_in=False)

    for index, length in enumerate(lengths):
        spoke = index // 48
        clockwise_spin = (index // 24) % 2
        spoke_index = index % 24

        distance_from_origin = (1 + spacing) * line_resolution * (1 + spoke_index) + center_spoke_length
        angle_from_origin = spoke_point_translation_angles[spoke]

        spoke_point = get_translated_point(middle, distance_from_origin, angle_from_origin)
        max_point = get_translated_point(middle, int(distance_from_origin * sqrt(3) / 2), angle_from_origin + (pi / 6)*(-1)**(clockwise_spin))

        max_distance = get_distance(spoke_point, max_point)
        end_point = get_translated_point(spoke_point, max_distance[0] * (length / 8), max_distance[1])

        d.line([spoke_point, end_point], fill=(0,0,0), width=line_resolution)

    return img

# image decoding functions

def decode_processed_image(img):
    spoke_point_translation_angles = [
        pi / 2,
        3 * pi / 2,
        5 * pi / 6,
        pi / 6,
        7 * pi / 6,
        11 * pi / 6
    ]

    middle = (image_size // 2, image_size // 2)
    spoke_points = [ get_translated_point(middle, total_spoke_length, angle) for angle in spoke_point_translation_angles ]

    # d = ImageDraw.Draw(img)

    lengths = []
    for index in range(2 * spoke_count * lines_per_spoke):
        spoke = index // 48
        clockwise_spin = (index // 24) % 2
        spoke_index = index % 24

        distance_from_origin = (1 + spacing) * line_resolution * (1 + spoke_index) + center_spoke_length
        angle_from_origin = spoke_point_translation_angles[spoke]

        spoke_point = get_translated_point(middle, distance_from_origin, angle_from_origin)
        max_point = get_translated_point(middle, int(distance_from_origin * sqrt(3) / 2), angle_from_origin + (pi / 6)*(-1)**(clockwise_spin))

        max_distance, angle = get_distance(spoke_point, max_point)

        for length in range(1, 8):
            end_point = get_translated_point(spoke_point, max_distance * (length / 8), angle)

            samples = [ img.getpixel(get_translated_point(end_point, i, angle)) for i in range(-int(max_distance / 24), int(max_distance / 24)) ]
            avg = sum(samples) / len(samples)

            # for i in range(-int(max_distance / 24), int(max_distance / 24)):
            #     pixel = get_translated_point(end_point, i, angle)
            #     d.point([pixel], fill=255 - img.getpixel(pixel))

            if avg >= 10 and avg <= 235:
                lengths.append(length)
                break
        else:
            lengths.append(0)

    # img.show()

    # RS decoding

    payload = decode_line_lengths(lengths)

    try:
        data = decode_payload(payload)[0].decode()
    except:
        print("RS decryption failed: too many errors", file=sys.stderr)
        return

    return data

def process_image(img):
    # Render image in black and white

    out = Image.new('I', img.size, 0xffffff)
    thresh = (0.5) * (sum(ImageStat.Stat(img).mean) / 3)
    fn = lambda x : 255 if x > thresh else 0
    out = img.convert('L').point(fn, mode='1')

    # Get spoke points

    opencvImage = cv2.cvtColor(numpy.array(out.convert('RGB')), cv2.COLOR_RGB2BGR)
    real_spoke_points = find_hexagons(opencvImage)

    if real_spoke_points == []:
        print("Snowflake not detected.")
        return

    # perspective transform real spoke_points to ideal spoke_points

    orientation_hex_angles = [
        -pi / 2,     # top
        -5 * pi / 6, # top left
        -3 * pi / 2,  # bottom
        -11 * pi / 6 # bottom right
    ]

    middle = (image_size // 2, image_size // 2)
    orientation_hexes = [ get_translated_point(middle, total_spoke_length, angle) for angle in orientation_hex_angles ]
    out = perspective_transform(out, real_spoke_points, orientation_hexes)

    out.resize((image_size, image_size))

    return out

def old_decode_image(filename):
    try:
        img = Image.open(filename)
    except:
        print("no such file.")
        return

    # Rendering image in black and white

    out = Image.new('I', img.size, 0xffffff)
    thresh = (0.5) * (sum(ImageStat.Stat(img).mean) / 3)
    fn = lambda x : 255 if x > thresh else 0
    out = img.convert('L').point(fn, mode='1')

    opencvImage = cv2.cvtColor(numpy.array(out.convert('RGB')), cv2.COLOR_RGB2BGR)
    cv_real_spoke_points = find_hexagons(opencvImage)

    # get crop bounds

    pixels = out.load()
    
    # left = 0
    # white_pixel_count = 0
    # for x in range(out.size[0]):
    #     for y in range(out.size[1]):
    #         if pixels[x, y] == 255:
    #             white_pixel_count += 1
    #     if (white_pixel_count / out.size[1]) >= 0.95:
    #         left = x
    #         break
    #     else:
    #         white_pixel_count = 0
    
    # right = out.size[0] - 1
    # white_pixel_count = 0
    # for x in range(out.size[0] - 1, -1, -1):
    #     for y in range(out.size[1]):
    #         if pixels[x, y] == 255:
    #             white_pixel_count += 1
    #     if (white_pixel_count / out.size[1]) >= 0.95:
    #         right = x
    #         break
    #     else:
    #         white_pixel_count = 0
    
    # up = 0
    # white_pixel_count = 0
    # for y in range(out.size[1]):
    #     for x in range(out.size[0]):
    #         if pixels[x, y] == 255:
    #             white_pixel_count += 1
    #     if (white_pixel_count / out.size[0]) >= 0.95:
    #         up = y
    #         break
    #     else:
    #         white_pixel_count = 0
    
    # down = out.size[1] - 1
    # white_pixel_count = 0
    # for y in range(out.size[1] - 1, -1, -1):
    #     for x in range(out.size[0]):
    #         if pixels[x, y] == 255:
    #             white_pixel_count += 1
    #     if (white_pixel_count / out.size[0]) >= 0.95:
    #         down = y
    #         break
    #     else:
    #         white_pixel_count = 0

    # out = out.crop((left, up, right, down))

    # find real spoke_points (openCV)

    opencvImage = cv2.cvtColor(numpy.array(out.convert('RGB')), cv2.COLOR_RGB2BGR)
    cv_real_spoke_points = find_hexagons(opencvImage)

    # find real spoke_points

    real_outer_spoke_points = []
    find = False

    for x in range(out.size[0]):
        for y in range(out.size[1] - 1, -1, -1):
            if pixels[x, y] == 0:
                real_outer_spoke_points.append((x, y))
                find = True
                break
        if find:
            find = False
            break

    for x in range(out.size[0] - 1, -1, -1):
        for y in range(out.size[1]):
            if pixels[x, y] == 0:
                real_outer_spoke_points.append((x, y))
                find = True
                break
        if find:
            find = False
            break

    for y in range(out.size[1]):
        for x in range(out.size[0]):
            if pixels[x, y] == 0:
                real_outer_spoke_points.append((x, y))
                find = True
                break
        if find:
            find = False
            break

    for y in range(out.size[1] - 1, -1, -1):
        for x in range(out.size[0]):
            if pixels[x, y] == 0:
                real_outer_spoke_points.append((x, y))
                find = True
                break
        if find:
            find = False
            break

    # perspective transform real spoke_points to ideal spoke_points

    outer_spoke_point_translation_angles = [
        7 * pi / 6,
        pi / 6,
        pi / 2,
        3 * pi / 2
    ]

    middle = (image_size // 2, image_size // 2)
    ideal_outer_spoke_points = [ get_translated_point(middle, total_spoke_length + endpoint_spoke_length, angle) for angle in outer_spoke_point_translation_angles ]
    out = perspective_transform(out, real_outer_spoke_points, ideal_outer_spoke_points)

    # rotate image to proper orientation

    spoke_point_translation_angles = [
        pi / 6,
        pi / 2,
        5 * pi / 6,
        3 * pi / 2,
        7 * pi / 6,
        11 * pi / 6
    ]

    spoke_points = [ get_translated_point(middle, total_spoke_length, angle) for angle in spoke_point_translation_angles ]
    right_top, top, left_top, left_bottom, bottom, right_bottom = spoke_points

    if out.getpixel(right_top) == 255 and out.getpixel(left_bottom) == 255:
        # rotate 60 degrees widdershins
        out = out.rotate(angle=60, fillcolor=255)
        pass
    elif out.getpixel(left_top) == 255 and out.getpixel(right_bottom) == 255:
        # rotate 60 degrees clockwise
        out = out.rotate(angle=300, fillcolor=255)
        pass

    if out.getpixel(left_bottom) == 255:
        # flip horizontally
        out = out.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
    elif out.getpixel(right_top) == 255:
        # flip vertically
        out = out.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
    elif out.getpixel(right_bottom) == 255:
        # flip both ways AKA rotate 180 degrees
        out = out.transpose(method=Image.Transpose.ROTATE_180)
        pass

    out.resize((image_size, image_size))

    # decode processed image

    decode_processed_image(out)

# external API

def text_to_snowcode(text):
    data = text.encode()

    if len(data) > data_size_limit:
        print(f"limit: { data_size_limit } character maximum", file=sys.stderr)
        return

    data += (b'\0') * (data_size_limit - len(data)) # null character padding

    payload = encode_payload(data)
    lengths = encode_line_lengths(payload)

    return build_image(lengths)

def snowcode_to_text(filename):
    try:
        img = Image.open(filename)
    except:
        print("no such file", file=sys.stderr)
        return

    processed_img = process_image(img)
    if not processed_img:
        return

    return decode_processed_image(processed_img)

def main():
    print("encoding or decoding?")
    option = input()
    if len(option) != 0 and option[0] in ['e', 'E']:
        print("encoding selected. provide payload:")
        img = text_to_snowcode(input())
        if img:
            img.show()
    else:
        print("decoding selected. provide filename:")
        text = snowcode_to_text(input())
        if text:
            print(text)

if __name__ == "__main__":
    main()

# To Do:
## Website for nice user interface (potentially)
## Variable encoding schemes for different payload lengths - version information stored in center hex