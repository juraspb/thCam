/*
По умолчанию все функции модуля графического процессора синхронны, т.е. текущий поток ЦП блокируется до завершения операции.

gpu::Stream является оболочкой для cudaStream_t и позволяет использовать асинхронный неблокирующий вызов. Вы также можете прочитать «Руководство по программированию CUDA C» для получения подробной информации об асинхронном параллельном выполнении CUDA.

Большинство функций модуля графического процессора имеют дополнительный параметр gpu::Stream. Если вы передадите поток, отличный от потока по умолчанию, вызов функции будет асинхронным, и вызов будет добавлен в очередь команд потока.

Также gpu::Stream предоставляет методы для асинхронной передачи памяти между CPU<->GPU и GPU<->GPU. Но CPU<->GPU асинхронная передача памяти работает только с памятью хоста с блокировкой страниц. Есть еще один класс gpu::CudaMem, который инкапсулирует такую ​​память.

В настоящее время могут возникнуть проблемы, если одна и та же операция дважды ставится в очередь с разными данными в разные потоки. Некоторые функции используют постоянную или текстурную память графического процессора, и следующий вызов может обновить память до того, как будет завершен предыдущий. Но асинхронный вызов различных операций безопасен, потому что каждая операция имеет свой собственный постоянный буфер. Операции копирования / выгрузки / загрузки / установки памяти в хранимые вами буферы также безопасны.

Вот небольшой образец:
*/

// allocate page-locked memory
CudaMem host_src_pl(768, 1024, CV_8UC1, CudaMem::ALLOC_PAGE_LOCKED);
CudaMem host_dst_pl;

// get Mat header for CudaMem (no data copy)
Mat host_src = host_src_pl;

// fill mat on CPU
someCPUFunc(host_src);

GpuMat gpu_src, gpu_dst;

// create Stream object
Stream stream;

// next calls are non-blocking

// first upload data from host
stream.enqueueUpload(host_src_pl, gpu_src);
// perform blur
blur(gpu_src, gpu_dst, Size(5,5), Point(-1,-1), stream);
// download result back to host
stream.enqueueDownload(gpu_dst, host_dst_pl);

// call another CPU function in parallel with GPU
anotherCPUFunc();

// wait GPU for finish
stream.waitForCompletion();

// now you can use GPU results
Mat host_dst = host_dst_