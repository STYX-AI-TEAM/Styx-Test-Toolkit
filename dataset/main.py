def load_any_dataset(dataset_name, subset_name=None, split='', rows):
dataset_map = {
    "CALM" : ""
}
if subset_name:
        data = load_dataset(dataset_name, subset_name)
    else:
        data = load_dataset(dataset_name)


