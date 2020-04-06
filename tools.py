import multiprocessing as mp


def agg_cal(func, *args, process_num):


    result_list1 = []
    result_list2 = []

    if not process_num:
        pool = mp.Pool()
    else:
        pool = mp.Pool(processes=process_num)

    for i in args:
        result1 = pool.apply_async(func, i)
        result_list1.append(result1)

    pool.close()
    pool.join()

    for r in result_list1:
        result2 = r.get()
        result_list2.append(result2)

    return result_list2
