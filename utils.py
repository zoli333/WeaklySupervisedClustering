import numpy as np


def perturb_labels(labels, psnr):
    # PERTURB_LABELS add psnr% of noise to every class in labels

    label_set = np.unique(labels)
    # print(label_set)
    perturbed_labels = labels.copy()

    for ilabel in range(len(label_set)):
        label = label_set[ilabel]
        idx = np.argwhere(labels == label)[:, 0]
        # print("idx", idx)
        n = len(idx)
        sample_set = np.setdiff1d(label_set, label)
        # print("sample set", sample_set)
        # Sample new class IDs to perturb
        num_perturb = int(np.fix(psnr * n))
        # print("num_perturb", num_perturb)
        # pos = np.random.randint(0, len(sample_set), size=num_perturb)
        # pos = np.random.choice(len(sample_set), min(num_perturb, len(sample_set)), replace=False)
        pos = np.random.choice(len(sample_set), num_perturb, replace=True)
        # print("pos", pos)
        perturb_class_ids = sample_set[pos]

        # Apply perturbation
        perturb_idx = np.random.permutation(n)[:num_perturb]
        perturbed_labels[idx[perturb_idx]] = perturb_class_ids
        # print("----------------")
    return perturbed_labels


if __name__ == '__main__':
    # Example usage:
    original_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    perturbed_labels = perturb_labels(original_labels, psnr=0.2)

    print("Original Labels:", original_labels)
    print("Perturbed Labels:", perturbed_labels)