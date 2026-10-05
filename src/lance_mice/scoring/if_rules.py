"""Adapted Edit-R2 IF rules at 26b55829246e1a67fc3c8d522324fee3d49cd954.
Only I/O injection, debug removal and exception propagation changed. See NOTICE.
"""
from __future__ import annotations
from typing import Tuple
from PIL import Image

def _calculate_iou(box1, box2):
    """Compute the IoU of two bounding boxes"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    if union == 0:
        return 0.0
    return intersection / union

def _count_object(boxes, iou_threshold=0.8):
    """Count the number of objects after de-duplication"""
    if len(boxes) == 0:
        return 0
    boxes = boxes.copy()
    while True:
        if len(boxes) <= 1:
            return len(boxes)
        boxes_removed = False
        i = 0
        while i < len(boxes):
            j = i + 1
            while j < len(boxes):
                if _calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                    boxes.pop(j)
                    boxes_removed = True
                else:
                    j += 1
            i += 1
        if not boxes_removed:
            return len(boxes)
VLM_ADD_PROMPT = 'The first image is the original, and the second image reflects the changes made according to the editing instruction in subject addition. Can you determine if the editing instruction was successfully applied?\nThe editing instruction is: {instruction}\n\nPlease respond with "yes" or "no.'
VLM_REPLACE_PROMPT = 'The first image is the original, and the second image reflects the changes made according to the editing instruction in subject replacement. Can you determine if the editing instruction was successfully applied?\nThe editing instruction is: {instruction}\n\nPlease respond with "yes" or "no.'
VLM_COLOR_PROMPT = "Look at the object in the image. Is the {object_name} {new_color}? Please answer only 'YES' or 'NO'."
VLM_MATERIAL_PROMPT = "Is it possible that the {object_name} is made of {new_material}? Please answer only 'YES' or 'NO'."
VLM_TEXT_PROMPT = 'What text do you see in this image? Output only the text content, nothing else.'
VLM_BACKGROUND_PROMPT = "Look at the background of this image. Does the background show [{background}]? Please answer only 'YES' or 'NO'."
VLM_STRICT_COUNT_PROMPT = "You are asked to count the number of an object. Please answer only the number. \nFor example, if there are 3 dogs, you should answer '3'. If there is no dog, you should answer '0'.\n\nObject name: {object_name}\nNumber of objects:"

def parse_instruction(task_type, instruction):
    """Parse the instruction and extract the key information"""
    import re
    if task_type == 'subject_replace':
        pattern = 'Replace \\[([^\\]]+)\\] with \\[([^\\]]+)\\]'
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1), match.group(2)]
    elif task_type == 'subject_remove':
        pattern = 'Remove \\[([^\\]]+)\\]'
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1)]
    elif task_type == 'material_alter':
        pattern = 'Change the material of \\[([^\\]]+)\\] to \\[([^\\]]+)\\]'
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1), match.group(2)]
    elif task_type == 'color_alter':
        pattern = 'Change the color of \\[([^\\]]+)\\] to \\[([^\\]]+)\\]'
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1), match.group(2)]
    elif task_type == 'subject_add':
        pattern1 = 'Add \\[([^\\]]+)\\] on the \\[([^\\]]+)\\] of \\[([^\\]]+)\\]'
        match1 = re.search(pattern1, instruction)
        if match1:
            return [match1.group(1), match1.group(2), match1.group(3)]
        pattern2 = 'Add \\[([^\\]]+)\\]'
        match2 = re.search(pattern2, instruction)
        if match2:
            return [match2.group(1)]
    elif task_type == 'text_change':
        pattern1 = "Replace the text '\\[([^\\]]+)\\]' on \\[([^\\]]+)\\] with '\\[([^\\]]+)\\]'"
        match1 = re.search(pattern1, instruction)
        if match1:
            return [match1.group(1), match1.group(2), match1.group(3)]
        pattern2 = "Add text '\\[([^\\]]+)\\]' on the image"
        match2 = re.search(pattern2, instruction)
        if match2:
            return [match2.group(1)]
    elif task_type == 'position_change':
        pattern = 'Change the position of \\[([^\\]]+)\\] to \\[([^\\]]+)\\] of \\[([^\\]]+)\\]'
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1), match.group(2), match.group(3)]
    elif task_type == 'count_change':
        pattern = 'Change the count of \\[([^\\]]+)\\] to \\[([^\\]]+)\\]'
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1), match.group(2)]
    elif task_type == 'background_change':
        pattern1 = 'Change the background to ([^,]+), remain the (.+) unchanged'
        match1 = re.search(pattern1, instruction)
        if match1:
            return [match1.group(1)] + match1.group(2).split(',')
        pattern2 = 'Change the background to \\[([^\\]]+)\\]'
        match2 = re.search(pattern2, instruction)
        if match2:
            return [match2.group(1)]
    return None

def evaluate_single(ref_img: Image.Image, gen_img: Image.Image, prompt: str, formatted_instruction: str, task_type: str, call_groundingdino, call_vlm, call_vlm_2image) -> Tuple[float, str]:
    """
    Evaluate instruction following for a single image

    Returns:
        (score, reason): score is 0 or 1
    """
    position_threshold = 0.03
    if task_type == 'subject_add':
        parsed = parse_instruction('subject_add', formatted_instruction)
        if parsed is None:
            return (0, f'Invalid instruction format: {formatted_instruction}')
        if len(parsed) == 1:
            new_object = parsed[0]
            vlm_resp = call_vlm_2image(ref_img, gen_img, VLM_ADD_PROMPT.format(instruction=formatted_instruction))
            if 'yes' not in vlm_resp:
                return (0, f"VLM pre-check failed. Response: '{vlm_resp}'")
            detections = call_groundingdino(gen_img, new_object, return_all=True)
            if len(detections.get('score', [])) > 0:
                return (1, f"Successfully added [{new_object}]. Detected {len(detections['score'])} instance(s).")
            else:
                return (0, f'Failed to add [{new_object}]. No instances detected.')
        elif len(parsed) == 3:
            new_object, position, ref_object = parsed
            vlm_resp = call_vlm_2image(ref_img, gen_img, VLM_ADD_PROMPT.format(instruction=formatted_instruction))
            if 'yes' not in vlm_resp:
                return (0, f"VLM pre-check failed. Response: '{vlm_resp}'")
            src_ref = call_groundingdino(ref_img, ref_object, return_all=True)
            target_obj = call_groundingdino(gen_img, new_object, return_all=True)
            if len(src_ref.get('score', [])) == 0 or len(target_obj.get('score', [])) == 0:
                return (0, f"Failed to detect objects. Ref: {len(src_ref.get('score', []))}, Target: {len(target_obj.get('score', []))}")
            ref_centers = src_ref.get('center', [])
            target_centers = target_obj.get('center', [])
            for ref_center in ref_centers:
                ref_norm = [ref_center[0] / ref_img.size[0], ref_center[1] / ref_img.size[1]]
                for target_center in target_centers:
                    target_norm = [target_center[0] / gen_img.size[0], target_center[1] / gen_img.size[1]]
                    if position == 'left' and target_norm[0] - ref_norm[0] < -position_threshold:
                        return (1, f'Successfully added [{new_object}] to the left of [{ref_object}]')
                    elif position == 'right' and target_norm[0] - ref_norm[0] > position_threshold:
                        return (1, f'Successfully added [{new_object}] to the right of [{ref_object}]')
                    elif position == 'above' and target_norm[1] - ref_norm[1] < -position_threshold:
                        return (1, f'Successfully added [{new_object}] above [{ref_object}]')
                    elif position == 'below' and target_norm[1] - ref_norm[1] > position_threshold:
                        return (1, f'Successfully added [{new_object}] below [{ref_object}]')
            return (0, f'Failed to add [{new_object}] to the {position} of [{ref_object}]')
    elif task_type == 'subject_remove':
        parsed = parse_instruction('subject_remove', formatted_instruction)
        if parsed is None:
            return (0, f'Invalid instruction format: {formatted_instruction}')
        object_name = parsed[0]
        src_obj = call_groundingdino(ref_img, object_name, threshold=0.35, delete_large_box=True)
        if len(src_obj.get('score', [])) == 0:
            return (0, f'Failed to detect [{object_name}] in source image')
        src_box = src_obj['box'][0]
        target_obj = call_groundingdino(gen_img, object_name, threshold=0.35, delete_large_box=True)
        if len(target_obj.get('score', [])) == 0:
            return (1, f'Successfully removed [{object_name}]')
        if _calculate_iou(src_box, target_obj['box'][0]) < 0.2:
            return (1, f'Successfully removed [{object_name}] (different location)')
        return (0, f'Failed to remove [{object_name}]')
    elif task_type == 'subject_replace':
        parsed = parse_instruction('subject_replace', formatted_instruction)
        if parsed is None:
            return (0, f'Invalid instruction format: {formatted_instruction}')
        object_name, new_object = parsed
        vlm_resp = call_vlm_2image(ref_img, gen_img, VLM_REPLACE_PROMPT.format(instruction=formatted_instruction))
        if 'yes' not in vlm_resp:
            return (0, f"VLM pre-check failed. Response: '{vlm_resp}'")
        object_name = object_name[:-2] if object_name.endswith('es') else object_name[:-1] if object_name.endswith('s') else object_name
        new_object = new_object[:-2] if new_object.endswith('es') else new_object[:-1] if new_object.endswith('s') else new_object
        src_obj = call_groundingdino(ref_img, object_name, return_all=True)
        target_obj = call_groundingdino(gen_img, new_object, return_all=True)
        if len(src_obj.get('score', [])) == 0 or len(target_obj.get('score', [])) == 0:
            return (0, f"Failed to detect objects. Source: {len(src_obj.get('score', []))}, Target: {len(target_obj.get('score', []))}")
        for src_box in src_obj.get('box', []):
            for target_box in target_obj.get('box', []):
                if _calculate_iou(src_box, target_box) > 0:
                    return (1, f'Successfully replaced [{object_name}] with [{new_object}]')
        return (0, f'Failed to replace [{object_name}] with [{new_object}]')
    elif task_type == 'color_alter':
        parsed = parse_instruction('color_alter', formatted_instruction)
        if parsed is None:
            return (0, f'Invalid instruction format: {formatted_instruction}')
        object_name, new_color = parsed
        vlm_resp = call_vlm_2image(ref_img, gen_img, f'Is the {object_name} now {new_color}? Answer YES or NO only.')
        if 'yes' in vlm_resp:
            return (1, f'Successfully changed color of [{object_name}] to [{new_color}]')
        else:
            return (0, f"Failed to change color of [{object_name}] to [{new_color}]. VLM: '{vlm_resp}'")
    elif task_type == 'material_alter':
        parsed = parse_instruction('material_alter', formatted_instruction)
        if parsed is None:
            return (0, f'Invalid instruction format: {formatted_instruction}')
        object_name, new_material = parsed
        vlm_resp = call_vlm_2image(ref_img, gen_img, f'Is the {object_name} now made of {new_material}? Answer YES or NO only.')
        if 'yes' in vlm_resp:
            return (1, f'Successfully changed material of [{object_name}] to [{new_material}]')
        else:
            return (0, f"Failed to change material of [{object_name}] to [{new_material}]. VLM: '{vlm_resp}'")
    elif task_type == 'text_change':
        parsed = parse_instruction('text_change', formatted_instruction)
        if parsed is None:
            return (0, f'Invalid instruction format: {formatted_instruction}')
        if len(parsed) == 3:
            existing_text, object_name, new_text = parsed
            src_obj = call_groundingdino(ref_img, object_name)
            if len(src_obj.get('score', [])) == 0:
                return (0, f'Failed to detect [{object_name}] in source image')
            vlm_resp = call_vlm(gen_img, VLM_TEXT_PROMPT)
            if new_text.lower() in vlm_resp:
                return (1, f"Successfully changed text to '[{new_text}]'. VLM detected: '{vlm_resp}'")
            else:
                return (0, f"Failed to change text to '[{new_text}]'. VLM detected: '{vlm_resp}'")
        elif len(parsed) == 1:
            new_text = parsed[0]
            vlm_resp = call_vlm(gen_img, VLM_TEXT_PROMPT)
            if new_text.lower() in vlm_resp:
                return (1, f"Successfully added text '[{new_text}]'. VLM detected: '{vlm_resp}'")
            else:
                return (0, f"Failed to add text '[{new_text}]'. VLM detected: '{vlm_resp}'")
    elif task_type == 'position_change':
        parsed = parse_instruction('position_change', formatted_instruction)
        if parsed is None:
            return (0, f'Invalid instruction format: {formatted_instruction}')
        target_object, position, reference_object = parsed
        src_ref = call_groundingdino(ref_img, reference_object, return_all=True, threshold=0.4)
        src_target = call_groundingdino(ref_img, target_object, return_all=True, threshold=0.4)
        target_ref = call_groundingdino(gen_img, reference_object, return_all=True, threshold=0.4)
        target_target = call_groundingdino(gen_img, target_object, return_all=True, threshold=0.4)
        src_ref_count = _count_object(src_ref.get('box', []))
        target_ref_count = _count_object(target_ref.get('box', []))
        src_target_count = _count_object(src_target.get('box', []))
        target_target_count = _count_object(target_target.get('box', []))
        if src_ref_count != target_ref_count:
            return (0, f'Reference object count changed from {src_ref_count} to {target_ref_count}')
        if src_target_count != target_target_count:
            return (0, f'Target object count changed from {src_target_count} to {target_target_count}')
        if len(target_ref.get('score', [])) > 0 and len(target_target.get('score', [])) > 0:
            ref_center = target_ref['center'][0]
            target_center = target_target['center'][0]
            if position == 'left' and target_center[0] < ref_center[0]:
                return (1, f'Successfully moved [{target_object}] to the left of [{reference_object}]')
            elif position == 'right' and target_center[0] > ref_center[0]:
                return (1, f'Successfully moved [{target_object}] to the right of [{reference_object}]')
            elif position == 'above' and target_center[1] < ref_center[1]:
                return (1, f'Successfully moved [{target_object}] above [{reference_object}]')
            elif position == 'below' and target_center[1] > ref_center[1]:
                return (1, f'Successfully moved [{target_object}] below [{reference_object}]')
            else:
                return (0, f'Failed to move [{target_object}] to the {position} of [{reference_object}]')
        return (0, f'Failed to detect objects in target image')
    elif task_type == 'count_change':
        parsed = parse_instruction('count_change', formatted_instruction)
        if parsed is None:
            return (0, f'Invalid instruction format: {formatted_instruction}')
        object_name, target_count = parsed
        src_obj = call_groundingdino(ref_img, object_name)
        target_obj = call_groundingdino(gen_img, object_name, return_all=True)
        actual_count = _count_object(target_obj.get('box', []))
        if len(src_obj.get('score', [])) > 0 and int(actual_count) == int(target_count):
            return (1, f'Successfully changed count of [{object_name}] to {target_count}')
        else:
            return (0, f'Failed to change count of [{object_name}] to {target_count}. Actual count: {actual_count}')
    elif task_type == 'background_change':
        parsed = parse_instruction('background_change', formatted_instruction)
        if parsed is None:
            return (0, f'Invalid instruction format: {formatted_instruction}')
        background = parsed[0]
        remain_objects = parsed[1:] if len(parsed) > 1 else []
        for obj in remain_objects:
            obj_det = call_groundingdino(gen_img, obj, threshold=0.25)
            if len(obj_det.get('score', [])) == 0:
                return (0, f'Failed to detect remaining object [{obj}]')
            box = obj_det['box'][0]
            if abs(box[2] - box[0]) > 0.9 and abs(box[3] - box[1]) > 0.9:
                return (0, f'False positive detection of [{obj}]')
        vlm_resp = call_vlm(gen_img, VLM_BACKGROUND_PROMPT.format(background=background))
        if 'yes' in vlm_resp:
            return (1, f'Successfully changed background to [{background}]')
        else:
            return (0, f"Failed to change background to [{background}]. VLM: '{vlm_resp}'")
    else:
        return (0, f'Unknown task type: {task_type}')
