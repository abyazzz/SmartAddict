<?php

    $dataset = "dataset yang di gunakan notebook/retrain model";
    $predict_users_session_table = "table sementara untuk menyimpan input users sebanyak 50 row data";
    $input_10_fitur_user = "fitur yang diinput user untuk di predik";
    $insert_db_data_users = "query untuk insert data" . $input_10_fitur_user ."ke table " . $predict_users_session_table;
    $insert_data_table_dataset = "Query insert data" . $predict_users_session_table . "ke" . $dataset . ";";
    
    //program jalan. user input fitur untuk predik di halamn predik

    if ($insert_db_data_users === true) {
        if ($predict_users_session_table === 49) {
            print('input');
        }
    }
 
?>