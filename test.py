# from datasets import load_dataset

# imdb = load_dataset("imdb")



from datasets import load_dataset

eli5 = load_dataset("eli5_category", split="train[:5000]")