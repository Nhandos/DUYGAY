import argparse



class DigitDetector(


def main(args):

    desc_database = []
    
    # Generate database of descriptor
    for image_path in args.images:
        image = cv2.imread(image_path)
        orb = cv2.ORB_create()

        kp = orb.detect(image, None)

        kp, des = orb.compute(img, kp)
        desc_database.append(des)

    for image_path in glob.glob('./data/BuidlingSignage/Train/*.jpg'):
        image.

if __name__ == '__main__':
    parser = argparse.argument_parser()
    parser.add_argument('images', type='str', nargs='+', help='image paths')
    args = parser.parse_args()

    main(args)
    
