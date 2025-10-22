def build_env_diff(reqs: str, satisfied_reqs: str):
    r_dict, s_dict = dict(), dict()

    with open(reqs, mode="r") as r:
        ls = r.readlines()
        for l in ls:
            info = l.split("==")
            if info:
                r_dict[info[0]] = l
    
    with open(satisfied_reqs, mode="r") as s:
        ls = s.readlines()
        for l in ls:
            info = l.split("==")
            if info:
                s_dict[info[0]] = l
    
    diff = []
    for k, v in r_dict.items():
        if k not in s_dict:
            diff.append(v)
    
    return diff


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", type=str, default="requirements.txt")
    parser.add_argument("-s", "--satisfied-reqs", type=str, default="satisfied_reqs.txt")

    args = parser.parse_args()

    diff = build_env_diff(args.r, args.satisfied_reqs)

    with open("new_requirements.txt", mode="w", encoding="utf-8") as f:
        f.writelines(diff)
    pass


if __name__ == "__main__":
    main()