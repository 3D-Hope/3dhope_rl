import cv2
import numpy as np
from scipy.ndimage import binary_dilation
import argparse
import numpy as np
import pickle
from threed_front.evaluation import ThreedFrontResults
from steerable_scene_generation.datasets.custom_scene.threed_front_encoding import get_dataset_raw_and_encoded
from steerable_scene_generation.datasets.custom_scene.custom_scene_final import update_data_file_paths
# from threed_front.evaluation.utils import count_out_of_boundary, compute_bbox_iou
import os
"""Script for calculating physcene metrics. """

def map_to_image_coordinate(point, scale, image_size):
        x, y = point
        x_image = int(x / scale * image_size/2)+image_size/2
        y_image = int(y / scale * image_size/2)+image_size/2
        return x_image, y_image
    
def calc_bbox_masks(bbox,class_labels,image,image_size,scale,robot_width,floor_plan_mask, box_wall_count): 
    """
    type: ["bbox","front_line", "front_center"]
    """
    save_path = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/tmp"
    os.makedirs(save_path, exist_ok=True)
    box_masks = []
    handle_points = []
    for box,class_label in zip(bbox,class_labels):
        box_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        box_wall_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        center = map_to_image_coordinate(box[:3][0::2], scale, image_size)
        size = (int(box[3] / scale * image_size / 2) * 2,
                int(box[5] / scale * image_size / 2) * 2)
        angle = box[-1]
        w, h = size
        open_size = 0
        rot_center = center
        handle = np.array([0, h/2+open_size+robot_width/2+1])
        bbox = np.array([[-w/2, -h/2],
                        [-w/2, h/2+open_size],
                        [w/2, h/2+open_size],
                        [w/2, -h/2],
                        ])
        
        R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

        box_points = bbox.dot(R) + rot_center
        handle_point = handle.dot(R) + rot_center
        handle_points.append(handle_point)
        box_points = np.intp(box_points)
        # box wall collision
        cv2.fillPoly(box_wall_mask, [box_points], (0, 255, 0))
        #cv2.imwrite(os.path.join(save_path, "debug3.png"), box_wall_mask)
        box_wall_mask = box_wall_mask[:,:,1]==255

        if (box_wall_mask*(255-floor_plan_mask)).sum()>0:
            box_wall_count+=1
        #cv2.imwrite(os.path.join(save_path, "debug4.png"), floor_plan_mask)

        #image connected region
        cv2.drawContours(image, [box_points], 0,
                        (0, 255, 0), robot_width)  # Green (BGR)
        cv2.fillPoly(image, [box_points], (0, 255, 0))
        
        # per box mask
        cv2.drawContours(box_mask, [box_points], 0,
                        (0, 255, 0), robot_width)  # Green (BGR)
        cv2.fillPoly(box_mask, [box_points], (0, 255, 0))
        st_element = np.ones((3, 3), dtype=bool)
        box_mask = binary_dilation((box_mask[:, :, 1].copy()==255).astype(image.dtype), st_element)
        box_masks.append(box_mask)
    return box_masks, handle_points, box_wall_count, image

def cal_walkable_metric(floor_plan_vertices, floor_plan_faces, floor_plan_centroid, bboxes, robot_width=0.01, calc_object_area=False):
    save_path = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/tmp"
    os.makedirs(save_path, exist_ok=True)
    vertices, faces = floor_plan_vertices, floor_plan_faces
    vertices = vertices - floor_plan_centroid
    vertices = vertices[:, 0::2]
    scale = np.abs(vertices).max()+0.2
    bboxes = bboxes[bboxes[:, 1] < 1.5]

    image_size = 256
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    robot_width = int(robot_width / scale * image_size/2)


    # draw face
    for face in faces:
        face_vertices = vertices[face]
        face_vertices_image = [
            map_to_image_coordinate(v, scale, image_size) for v in face_vertices]

        pts = np.array(face_vertices_image, np.int32)
        pts = pts.reshape(-1, 1, 2)
        color = (255, 0, 0)  # Blue (BGR)
        cv2.fillPoly(image, [pts], color)

    kernel = np.ones((robot_width, robot_width))
    image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)
    # draw bboxes
    #cv2.imwrite(os.path.join(save_path, "image1.png"), image)
    for box in bboxes:
        center = map_to_image_coordinate(box[:3][0::2], scale, image_size)
        size = (int(box[3] / scale * image_size / 2) * 2,
                int(box[5] / scale * image_size / 2) * 2)
        angle = box[-1]

        # calculate box vertices
        box_points = cv2.boxPoints(
            ((center[0], center[1]), size, -angle/np.pi*180))
        box_points = np.intp(box_points)

        cv2.drawContours(image, [box_points], 0,
                         (0, 255, 0), robot_width)  # Green (BGR)
        cv2.fillPoly(image, [box_points], (0, 255, 0))

    #cv2.imwrite(os.path.join(save_path, "image2.png"), image)

    if calc_object_area:
        green_cnt = 0
        blue_cnt = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if list(image[i][j]) == [0, 255, 0]:
                    green_cnt += 1
                elif list(image[i][j]) == [255, 0, 0]:
                    blue_cnt += 1
        object_area_ratio = green_cnt/(blue_cnt+green_cnt)
        
    walkable_map = image[:, :, 0].copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        walkable_map, connectivity=8)


    walkable_map_max = np.zeros_like(walkable_map)
    for label in range(1, num_labels):
        mask = np.zeros_like(walkable_map)
        mask[labels == label] = 255

        if mask.sum() > walkable_map_max.sum():
            # room connected component with door
            walkable_map_max = mask.copy()

        # print("walkable_rate:", walkable_map_max.sum()/walkable_map.sum())
        if calc_object_area:
            return walkable_map_max.sum()/walkable_map.sum(), object_area_ratio
        else:
            return walkable_map_max.sum()/walkable_map.sum()
    if calc_object_area:    
        return 0.,object_area_ratio
    else:
        return 0.
    
def calc_wall_overlap(threed_front_results, raw_dataset, encoded_dataset, cfg, robot_real_width=0.3,calc_object_area=False,classes=None):
    save_path = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/tmp"
    os.makedirs(save_path, exist_ok=True)
    # print(f"Scene layout: {scene_layout}, GT scene layout: {gt_scene_layout}")
    # print(f"BBox: {bbox}, Class Labels: {class_labels}")
    # print(f"Floor Plan Vertices: {floor_plan_vertices}, Floor Plan Faces: {floor_plan_faces}, Floor Plan Centroid: {floor_plan_centroid}")
    # import sys
    # sys.exit()
    
    box_wall_count = 0
    accessable_count = 0
    box_count = 0
    walkable_metric_list = []
    accessable_rate_list = []
    # from tqdm import tqdm
    
        # for scene_idx, scene_layout in threed_front_results:
    
    
    # print(f"Scene {scene_idx} - Floor Plan Vertices: {floor_plan_vertices}, Floor Plan Faces: {floor_plan_faces}, Floor Plan Centroid: {floor_plan_centroid}")
    # print(f"Scene {scene_idx} - Class Labels: {class_labels}, BBox: {bbox}")
    # break

    # for i in tqdm(range(len(synthesized_scenes))):
    from tqdm import tqdm
    for scene_idx, scene_layout in tqdm(threed_front_results):
        raw_item = encoded_dataset[scene_idx]
        # print(f"Scene {scene_idx} - Raw Item: {raw_item.keys()}")
        
        floor_plan_vertices = raw_item["floor_plan_vertices"]
        floor_plan_faces = raw_item["floor_plan_faces"]
        floor_plan_centroid = raw_item["floor_plan_centroid"]
        
        # print(scene_layout["class_labels"].shape, scene_layout["translations"].shape, scene_layout["sizes"].shape, scene_layout["angles"].shape)
        valid_idx = np.ones_like(scene_layout["class_labels"][:,0], dtype=bool)
        class_labels = scene_layout["class_labels"][valid_idx]
 
        bbox = np.concatenate([
                    scene_layout["translations"][valid_idx],
                    scene_layout["sizes"][valid_idx],
                    scene_layout["angles"][valid_idx],
                ],axis=-1)

            
        vertices, faces = floor_plan_vertices, floor_plan_faces
        vertices = vertices - floor_plan_centroid
        vertices = vertices[:, 0::2]
        # vertices = vertices[:, :2]
        scale = np.abs(vertices).max()+0.2

        image_size = 256
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        robot_width = int(robot_real_width / scale * image_size/2)

        
        
        # draw face
        for face in faces:
            face_vertices = vertices[face]
            face_vertices_image = [
                map_to_image_coordinate(v,scale, image_size) for v in face_vertices]

            pts = np.array(face_vertices_image, np.int32)
            pts = pts.reshape(-1, 1, 2)
            color = (255, 0, 0)  # Blue (BGR)
            cv2.fillPoly(image, [pts], color)
        
        floor_plan_mask = (image[:,:,0]==255)*255
        #cv2.imwrite(os.path.join(save_path, "debug_floor.png"), floor_plan_mask)
        # 缩小墙边界，机器人行动范围
        kernel = np.ones((robot_width, robot_width))
        image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)
        
        box_masks, handle_points, box_wall_count, image = calc_bbox_masks(bbox,class_labels,image,image_size,scale,robot_width,floor_plan_mask, box_wall_count)
        #cv2.imwrite(os.path.join(save_path, "debug.png"), image)
        # breakpoint()
        walkable_map = image[:, :, 0].copy()
        #cv2.imwrite(os.path.join(save_path, "debug2.png"), walkable_map)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            walkable_map, connectivity=8)
        # 遍历每个连通域

        accessable_rate = 0
        for label in range(1, num_labels):
            mask = np.zeros_like(walkable_map)
            mask[labels == label] = 1
            accessable_count = 0
            for box_mask in box_masks:
                if (box_mask*mask).sum()>0:
                    accessable_count += 1
            accessable_rate += accessable_count/len(box_masks)*mask.sum()/(labels!=0).sum()
        accessable_rate_list.append(accessable_rate)
        box_count += len(box_masks)

        #walkable map area rate
        if calc_object_area:
            walkable_rate, object_area_ratio = cal_walkable_metric(floor_plan_vertices, floor_plan_faces, floor_plan_centroid, bbox, robot_width=0.3,calc_object_area=True)
        else:
            walkable_rate = cal_walkable_metric(floor_plan_vertices, floor_plan_faces, floor_plan_centroid, bbox, robot_width=0.3)
        walkable_metric_list.append(walkable_rate)

    walkable_average_rate = sum(walkable_metric_list)/len(walkable_metric_list)
    accessable_rate = sum(accessable_rate_list)/len(accessable_rate_list)
    # accessable_handle_rate = sum(accessable_handle_rate_list)/len(accessable_handle_rate_list)
    box_wall_rate = box_wall_count/box_count
    
    print('walkable_average_rate:', walkable_average_rate)
    print('accessable_rate:', accessable_rate)
    # print('accessable_handle_rate:', accessable_handle_rate)
    print('box_wall_rate:', box_wall_rate)
    if calc_object_area:
        print('object_area_ratio:', object_area_ratio)
        return walkable_average_rate, accessable_rate, box_wall_rate, object_area_ratio
    else:
        return walkable_average_rate, accessable_rate, box_wall_rate

def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the FID scores between the real and the "
                     "synthetic images")
    )
    parser.add_argument(
        "result_file",
        help="Path to a pickled result file (ThreedFrontResults object)"
    )

    args = parser.parse_args(argv)

    # Load saved results
    with open(args.result_file, "rb") as f:
        threed_front_results = pickle.load(f)
    assert isinstance(threed_front_results, ThreedFrontResults)
    assert threed_front_results.floor_condition

    # Load dataset
    config = threed_front_results.config
    raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
            update_data_file_paths(config["data"], config),
            split=config["validation"].get("splits", ["test"]),
            max_length=config["network"]["sample_num_points"],
            include_room_mask=True
        )
    
    
    walkable_average_rate, accessable_rate, box_wall_rate = calc_wall_overlap(threed_front_results, raw_dataset, encoded_dataset, config, robot_real_width=0.3,calc_object_area=False,classes=None)

# Walkable Average Rate(walkable_average_rate): Path exists in the scene for the robot to walk through
# Accessable Rate(accessable_rate): The robot can access every object in the scene
# Box Wall Rate(box_wall_rate): Objects in scene are going out of bounds or not
if __name__ == "__main__":
    main(None)