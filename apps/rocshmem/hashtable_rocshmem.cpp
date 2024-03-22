#include "xmr.hpp"
#include "commons/mpi.hpp"
#include "commons/rocshmem.hpp"

#define MAX_AVG_COLLISIONS 4
#define HTABLE_FILL_RATIO 0.001

__device__ int clockrate;


int64_t TSIZE;

int insert_collisions = 0;
int delete_collisions = 0;
int inserts = 0;
int deletes = 0;

extern "C" {
  typedef struct{
    int64_t value;
    int64_t next;
  } t_elem;
}

typedef struct {
    MPI_Win twin; // table window
    MPI_Win lwin; // last entry window
    MPI_Win nwin; // nextfree window
  t_elem *table; // the table + heap
    int  nextfree; // the next free item in heap
  int64_t size; // size of the heap
  int64_t *last; // direct link to last element -- avoid traversing list!
  int p, r;
  int64_t tsize; // size of table
} t_hash;

__device__ int64_t hashfunc(int64_t val, int64_t d_tsize) {
  return val & (d_tsize-1);
}

void printHTAll(t_hash* hash, int n) {

  int val = -1;
  int nxt = -1;
  int owner = 0;
  int offset = 0;
  fprintf(stderr,"[%i] @@@@@@ Table: \n ",hash->r);
  for(int i = 0; i < TSIZE; i++) {
    if(i % hash->p == 0 && i > 0) {
      owner++;
      offset = 0;
      fprintf(stderr,"--\n");
    }

    MPI_Get(&val,1,MPI_INT,owner,offset,1,MPI_INT,hash->twin);
    MPI_Win_flush(owner, hash->twin);
    MPI_Get(&nxt,1,MPI_INT,owner,offset+1,1,MPI_INT,hash->twin);
    MPI_Win_flush(owner, hash->twin);
    fprintf(stderr,">> [%i] %i : (value: %i, next: %i)\n",owner,i,val,nxt);
    offset += 2;
  }

  owner = 0;
  offset = 0;

  fprintf(stderr,"\n[%i] @@@ Heap: \n",hash->r);
  for(int i = 0; i < n; i++) {

    if(i % hash->p == 0 && i > 0) {
      owner++;
      offset = 0;
      fprintf(stderr,"--\n");
    }

    MPI_Get(&val,1,MPI_INT,owner,hash->tsize*2 + offset,1,MPI_INT,hash->twin);
    MPI_Win_flush(owner, hash->twin);
    MPI_Get(&nxt,1,MPI_INT,owner,hash->tsize*2 + offset+1,1,MPI_INT,hash->twin);
    MPI_Win_flush(owner, hash->twin);
    fprintf(stderr,">> [%i] %li : (value: %i, next: %i)\n",owner,i+TSIZE,val,nxt);
    offset += 2;

  }
}

void printHTLocal(t_hash* hash) {
  printf("[%i] @@@@@@ Table: \n ",hash->r);
  for(int i = 0; i < hash->tsize; i++) {
    printf("[%i] %i : (value: %li, next: %li)\n",hash->r,i,(hash->table)[i].value, (hash->table)[i].next);
  }

  printf("\n[%i] @@@ Heap: \n",hash->r);
  for(int i = hash->tsize; i < hash->tsize+hash->size; i++) {
    printf("[%i] %i : (value: %li, next: %li)\n",hash->r,i,(hash->table)[i].value, (hash->table)[i].next);
  }
}

__global__ void check_val(int mype, int tot_num, int64_t* hash_d_table, int64_t* hash_d_last, int * hash_d_nextfree, int64_t d_tsize, int64_t d_size, int64_t* d_val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid==0){
        int cnt=0;
        int heapcnt=0;
        for(int i=0; i<d_tsize*2; i=i+2) {
            int pos = i;
            if(hash_d_table[pos] != -1) {
                cnt++;
                if(hash_d_table[pos+1] != -1) {
                    heapcnt++;
                    pos=hash_d_table[pos+1];
                    while(hash_d_table[pos+1] != -1) {
                        pos = hash_d_table[pos+1];
                        heapcnt++;
                    }
                }
            }
        }


        int cnt_1 = 0;
        for(int i = 0; i < (d_tsize+d_size)*2;i=i+2) {
            if(hash_d_table[i] != -1) {
                cnt_1++;
            }
        }
        if((cnt+heapcnt)== cnt_1) {
           if(mype==0) printf("ASSERT COUNT WRONG\n");
        };

    }
}
__global__ void insert(int mype, int tot_num, int64_t* hash_d_table, int64_t* hash_d_last, int * hash_d_nextfree, int64_t d_tsize, int64_t d_size, int64_t* d_val) {
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    roc_shmem_wg_init();
    int my_num=(tot_num)/1024;
    int my_start=tid*my_num;
    //printf(" Enter (%d,%d) tot_num=%d,my_num=%d, mod=%d\n",mype,tid, tot_num, my_num, tot_num%32);
    if ( tot_num%1024!=0){
        int delta=tot_num%1024;
        if (tid<delta) {
            my_num+=1;
            my_start=tid*my_num;
            //printf(" (%d,%d) delta=%d,my_num=%d,my_start=%d\n",mype,tid, delta, my_num, my_start);
        }else{
            if (tot_num>1024) tid=delta*(my_num+1)+(tid-delta)*my_num;
        }

    }

    if (my_num==0) return;
    //printf("(%d,%d) my_num=%d,size=%ld,%ld\n",mype,tid, my_num, d_tsize,d_size);
    int64_t elem;
    int64_t pos;
    int owner;
    for(int m = 0; m < my_num; m++) {
        elem = d_val[my_start+m];
        //printf("(%d,%d) elem=%ld\n",mype,tid,elem);
        pos = hashfunc(elem, d_tsize);
        //printf("(%d,%d) pos=%ld\n",mype,tid,pos);
        owner = (int) pos/d_tsize;
        //printf("(%d,%d) owner=%d\n",mype,tid,owner);

        // CAS if table position is available
        int64_t compare = -1;
        int64_t result = -2;
        int64_t offset /* in ints */ = (pos-owner*d_tsize)*2;

        result=roc_shmem_int64_atomic_compare_swap(&hash_d_table[offset],compare, elem, owner);
        //printf("%d, here1\n", mype);
        //MPI_Compare_and_swap(&elem, &compare, &result, MPI_INT64_T, owner, /* target disp */ offset, hash->twin);
        //MPI_Win_flush_local(owner, hash->twin);

        if(result != compare) { // we lost the CAS, no direct add

          int64_t offset = (pos-owner*d_tsize);
          int64_t origin = 1; // add 1
          int64_t result = -1; // address of newly added record

          // grab remote location
          result=roc_shmem_int_atomic_fetch_inc(&hash_d_nextfree[0],owner);
          //printf("%d, here2\n", mype);
          //MPI_Fetch_and_op(/* origin */ &origin, /* result */ &result, MPI_INT64_T, owner, /* disp */ 0, MPI_SUM, hash->nwin);
          //MPI_Win_flush_local(owner, hash->nwin);

          if(result >= d_tsize+d_size) {
            printf("Failed to insert due to the OVERFLOW\n");
            return;
          }

          // put element in grabbed location
          //MPI_Put(&elem, 1, MPI_INT64_T, owner, 2*(result), 1, MPI_INT64_T, hash->twin);

          // FIXME: @Nan int64_t is not available in roc_shmem_X_put so typecasting to long and using that. Please ensure this is correct.
          long *ptr = (long *)(&hash_d_table[2*(result)]);
          roc_shmem_long_put(ptr, &elem, 1, owner);

          //printf("%d, here3\n", mype);
          // try to put directly as table link
          int64_t newOffset = result;

          int64_t oldOffset = -111;    // mine
          offset = (pos-owner*d_tsize);

          // grab lastptr element and plug in ptr to new element
          oldOffset=roc_shmem_int64_atomic_swap(&hash_d_last[offset],newOffset,owner);
          //MPI_Fetch_and_op(/* origin */ &newOffset, /* result */ &oldOffset, MPI_INT64_T, owner, /* disp */ offset, MPI_REPLACE, hash->lwin);
          //MPI_Win_flush_local(owner, hash->lwin);

          //printf("%d, here4\n", mype);
          compare = -1;
          result = -2;
          offset /* in ints */ = (pos-owner*d_tsize)*2+1;

          // try to update table pointer directly (if it's the first update)
          result=roc_shmem_int64_atomic_compare_swap(&hash_d_table[offset],compare, newOffset, owner);
          //MPI_Compare_and_swap(&newOffset, &compare, &result, MPI_INT64_T, owner, /* target disp */ offset, hash->twin);
          //MPI_Win_flush_local(owner, hash->twin);
          //printf("%d, here5\n", mype);
          if(result != compare) { // we lost the CAS for the direct pointer -- no direct access to table!
            roc_shmem_long_put(&hash_d_table[2*(oldOffset)+1], &newOffset, 1, owner);
            //MPI_Put(&newOffset, 1, MPI_INT64_T, owner, 2*(oldOffset)+1, 1, MPI_INT64_T, hash->twin);
            //MPI_Win_flush(owner, hash->twin);
          }
        }
    }
}


__global__ void init_buffer(int64_t* hash_d_table,int64_t* hash_d_last,int* hash_d_nextfree,int64_t hash_tsize,int64_t hash_size){
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (tid==0){
        hash_d_nextfree[0] =(int) hash_tsize;
        for(int i=0; i<2*(hash_tsize+hash_size); i++) {
          if(i<hash_tsize) hash_d_last[i]=-1;
          hash_d_table[i]=-1;
        }
    }
    //for(int i=0; i<2*(hash_tsize+hash_size); i++) {
    //  if(i<hash_tsize) hash_last[i]=-1;
    //  hash_table[i]=-1;
    //}
    //printf("%d,done init\n",rank);
    //fflush(stdout);

}
int main(int c, char *v[]) {
    int rank, nranks;
    int mype, npes, mype_node, ndevices;

    MPI_CHECK(MPI_Init(&c, &v));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));


    auto mpi_comm = MPI_COMM_WORLD;
    //attr.mpi_comm = &mpi_comm;
    //nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    roc_shmem_init();
    mype = roc_shmem_my_pe();
    npes = roc_shmem_n_pes();

    char name[MPI_MAX_PROCESSOR_NAME]; // = "ROC_SHMEM HashTable";
    int resultlength;
    MPI_Get_processor_name(name, &resultlength);

    // application picks the device each PE will use
    auto status = hipGetDeviceCount(&ndevices);
    status = hipSetDevice(mype%ndevices);

    int get_cur_dev;
    status = hipGetDevice(&get_cur_dev);

    //cudaDeviceProp prop;
    //cudaGetDeviceProperties(&prop,0));
    //cudaMemcpyToSymbol(clockrate, (void *) &prop.clockRate, sizeof(int), 0,
    //                              cudaMemcpyHostToDevice));

    int LV = atoi(v[1]);
    printf("!!!!! roc_shmem %d/%d, ndevices=%d,cur=%d, node=%s, LV=%d\n", mype, npes, ndevices, get_cur_dev, name, LV);
    fflush(stdout);

    TSIZE = (int64_t)LV*(int64_t)npes;
    srand(rank*1000000);

    int64_t hash_tsize = LV;
    int64_t hash_size = (MAX_AVG_COLLISIONS)*LV;

    //printf("%d,size=%ld,%ld (int64_t)=%d\n",rank,hash_tsize,hash_size,sizeof(int64_t));
    //fflush(stdout);

    int64_t *hash_d_table, *hash_d_last;
    int *hash_d_nextfree;
    hash_d_table = (int64_t *)roc_shmem_malloc(sizeof(int64_t)*2*(hash_tsize+hash_size));
    hash_d_last = (int64_t *)roc_shmem_malloc(sizeof(int64_t)*hash_tsize);
    hash_d_nextfree = (int*)roc_shmem_malloc(sizeof(int)*1);
    //printf("%d,done nvshmem malloc,tot_size=%f MB\n",rank,(sizeof(int64_t)*2*(hash_tsize+hash_size)+sizeof(int64_t)*hash_tsize)/1e6);
    //fflush(stdout);

    //MPI_Barrier(MPI_COMM_WORLD);
    status = hipDeviceSynchronize();
    init_buffer<<<1,32>>>(hash_d_table,hash_d_last,hash_d_nextfree,hash_tsize,hash_size);
    status = hipGetLastError();
    status = hipDeviceSynchronize();
    //MPI_Barrier(MPI_COMM_WORLD);
    //printf("%d,done init=%d,%d\n",rank,hash_tsize,hash_size);
    //fflush(stdout);
    double t_start = 0.0;
    double t_end = 0.0;
    int nr_of_ins = HTABLE_FILL_RATIO*LV;
    int warmups = 0.1*nr_of_ins;
    double ins_time = 0.0;

    srand(mype*1000000);
    int64_t *val,*d_val;
    val=(int64_t*)malloc(sizeof(int64_t)*(nr_of_ins));

    for (int i=0;i<nr_of_ins;i++){
        val[i]=bigRandVal();
    }
    status = hipMalloc((void**)&d_val, sizeof(int64_t)*(nr_of_ins));
    status = hipMemcpy(d_val,  val, sizeof(int64_t)*(nr_of_ins), hipMemcpyHostToDevice);

    //MPI_Barrier(MPI_COMM_WORLD);
    status = hipDeviceSynchronize();
    //put everything on GPU
    insert<<<1,512,0,0>>>(mype,warmups,hash_d_table,hash_d_last,hash_d_nextfree, hash_tsize, hash_size,d_val);
    status = hipGetLastError();
    status = hipDeviceSynchronize();

    //MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();
    insert<<<1,1024,0,0>>>(mype,nr_of_ins,hash_d_table,hash_d_last,hash_d_nextfree, hash_tsize, hash_size,d_val);
    status = hipGetLastError();
    status = hipDeviceSynchronize();
    //MPI_Barrier(MPI_COMM_WORLD);
    t_end = MPI_Wtime();
    ins_time = t_end - t_start;

    check_val<<<1,1>>>(mype,nr_of_ins,hash_d_table,hash_d_last,hash_d_nextfree, hash_tsize, hash_size,d_val);
    status = hipGetLastError();
    status = hipDeviceSynchronize();


    if(mype == 0) {
      printf("%i\t%f\n",npes,ins_time);
    }

  roc_shmem_finalize();

  MPI_Finalize();
}
