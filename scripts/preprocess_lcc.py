import pandas as pd
import json
import re
from xml.dom.minidom import parse
from html import unescape

DEBUG = False


def remove_tags(xml_tag: str):
    return unescape(re.sub(r'<[a-zA-Z/]+>', '', xml_tag).strip())


def get_conceptual_metaphor(lm_instance, threshold):
    id = lm_instance.getAttribute("id")
    text_content = lm_instance.getElementsByTagName("TextContent")[0]
    annotations = lm_instance.getElementsByTagName("Annotations")[0]
    cm_metaphoricity_annotations = annotations.getElementsByTagName(
        "MetaphoricityAnnotations")[0]
    metaphoricity_annotation_tags = cm_metaphoricity_annotations.getElementsByTagName(
        "MetaphoricityAnnotation")
    if len(metaphoricity_annotation_tags) == 0:
        if DEBUG:
            print(
                f'lm_instance {id} doesnt have metaphoricity_annotation tags')
        return None

    metaphoricity_annotation = metaphoricity_annotation_tags[0]
    if float(metaphoricity_annotation.getAttribute(
            "score")) < float(threshold):
        if DEBUG:
            print(
                f'lm_instance {id} doesnt have metaphoricity annotations with score above {threshold}')
        return None

    curr = text_content.getElementsByTagName("Current")[0]
    cm_source_annotations = annotations.getElementsByTagName(
        "CMSourceAnnotations")
    if len(cm_source_annotations) == 0:
        if DEBUG:
            print(f'lm_instance {id} doesnt have cm_source_annotations')
        return None

    source_annotations = cm_source_annotations[0].getElementsByTagName(
        "CMSourceAnnotation")
    relevant_annotations = list(
        filter(
            lambda x: float(
                x.getAttribute("score")) >= threshold,
            source_annotations))
    if len(relevant_annotations) == 0:
        if DEBUG:
            print(
                f'lm_instance {id} doesnt have cm_source_annotations with score above/equal {threshold}')
        return None

    return dict(
        source_domain=','.join(list(set(map(lambda x: x.getAttribute("sourceConcept"), relevant_annotations)))),
        target_domain=lm_instance.getAttribute("targetConcept"),
        source_lexeme=remove_tags(curr.getElementsByTagName("LmSource")[0].toxml()),
        target_lexeme=remove_tags(curr.getElementsByTagName("LmTarget")[0].toxml()),
        sentence=remove_tags(text_content.getElementsByTagName("Current")[0].toxml()),
    )


def main():
    en_dataset = "../Data/LCC_Metaphor_Dataset.full/en_large.xml"
    with open(en_dataset, mode="r") as f:
        xml = parse(f)
        lm_instances = xml.getElementsByTagName(
            "LCC-Metaphor-LARGE")[0].getElementsByTagName("LmInstance")
        print(f'lm instances found: {len(lm_instances)}')
        cm_with_lexemes_and_annotations = list(filter(lambda x: x is not None, list(
            map(lambda x: get_conceptual_metaphor(x, 3), lm_instances))))
        print(
            f'cm with lexemes and annotations found {len(cm_with_lexemes_and_annotations)}')
        with open('./cms.json', 'w') as g:
            json.dump(cm_with_lexemes_and_annotations, g, indent=4)
        df = pd.DataFrame.from_records(cm_with_lexemes_and_annotations)
        print(df.head())
        df.to_csv('./cms.csv', sep=';')
    return None


if __name__ == "__main__":
    main()
