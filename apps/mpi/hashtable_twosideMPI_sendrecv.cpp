//
// Created by NanDing on 4/7/23.
//
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MAX_AVG_COLLISIONS 4
#define HTABLE_FILL_RATIO 0.001

int64_t bigRandVal() {

    int64_t random =
            (((int64_t) rand() <<  0) & 0x000000000000FFFFull) |
            (((int64_t) rand() << 16) & 0x00000000FFFF0000ull) |
            (((int64_t) rand() << 32) & 0x0000FFFF00000000ull) |
            (((int64_t) rand() << 48) & 0x0FFF000000000000ull);
    return random;

}

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
    int64_t nextfree; // the next free item in heap
    int64_t size; // size of the heap
    int64_t *last; // direct link to last element -- avoid traversing list!
    int p, r;
    int64_t tsize; // size of table
} t_hash;

int64_t hashfunc(int64_t val) {
    return val & (TSIZE-1);
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
        fprintf(stderr,">> [%i] %i : (value: %i, next: %i)\n",owner,i+TSIZE,val,nxt);
        offset += 2;

    }
}

void printHTLocal(t_hash* hash) {
    printf("[%i] @@@@@@ Table: \n ",hash->r);
    for(int i = 0; i < hash->tsize; i++) {
        printf("[%i] %i : (value: %i, next: %i)\n",hash->r,i,(hash->table)[i].value, (hash->table)[i].next);
    }

    printf("\n[%i] @@@ Heap: \n",hash->r);
    for(int i = hash->tsize; i < hash->tsize+hash->size; i++) {
        printf("[%i] %i : (value: %i, next: %i)\n",hash->r,i,(hash->table)[i].value, (hash->table)[i].next);
    }
}

void insert(int64_t* recv_bigRandVal,t_hash* hash) {
    int64_t elem = recv_bigRandVal[0];
    int64_t pos = recv_bigRandVal[1] ;
    int64_t owner = recv_bigRandVal[2];

    { // local update
        // CAS if table position is available
        int64_t offset /* in ints */ = (pos-owner*hash->tsize);
        int64_t compare = -1;
        if (hash->table[offset].value == compare) {
            hash->table[offset].value == elem;
        }else { // we lost the CAS, no direct add
            int64_t offset = (pos-owner*hash->tsize);
            int64_t result = hash->nextfree; // address of newly added record
            hash->nextfree+=1;
            if(result >= hash->tsize+hash->size) {
                fprintf(stderr,"Failed to insert due to the OVERFLOW\n");
                return;
            }
            hash->table[result].value=elem;
	    int64_t newOffset=result;
	    int64_t oldOffset=-111;
            offset = (pos-owner*hash->tsize);
            oldOffset=hash->last[offset];
	    
            hash->last[offset] = newOffset;

            offset /* in ints */ = (pos-owner*hash->tsize);
            int64_t compare = -1;
	    result=-2;
            if (hash->table[offset].next == compare) {
                hash->table[offset].next == newOffset;
            }else{
                hash->table[oldOffset].next == newOffset;
            }
        }
    }
}

int count(t_hash *hash) {

    int cnt=0;
    int heapcnt=0;
    for(int i=0; i<hash->tsize; i++) {
        int pos = i;
        if(hash->table[pos].value != -1) {
            cnt++;
            if(hash->table[pos].next != -1) {
                heapcnt++;
                pos=hash->table[pos].next;
                while(hash->table[pos].next != -1) {
                    pos = hash->table[pos].next;
                    heapcnt++;
                }
            }
        }
    }

    return cnt+heapcnt;
}

int countSimple(t_hash* hash) {
    int cnt = 0;
    for(int i = 0; i < hash->tsize+hash->size;i++) {
        if(hash->table[i].value != -1) {
            cnt++;
        }
    }
    return cnt;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    t_hash hash;
    int LV = atoi(argv[1]);

    MPI_Comm_size(MPI_COMM_WORLD, &hash.p);
    MPI_Comm_rank(MPI_COMM_WORLD, &hash.r);

    TSIZE = (int64_t)LV*(int64_t)hash.p;
    srand(hash.r*1000000);

    hash.tsize = LV;
    hash.size = (MAX_AVG_COLLISIONS)*LV;

    //int MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr)
    MPI_Alloc_mem((hash.tsize+hash.size)*sizeof(t_elem),MPI_INFO_NULL, &hash.table);
    MPI_Alloc_mem(hash.tsize*sizeof(int64_t),MPI_INFO_NULL, &hash.last);
    hash.nextfree = hash.tsize; // next free element in heap

    // initialize table
    for(int i=0; i<hash.tsize+hash.size; ++i) {
        if(i<hash.tsize) hash.last[i]=-1;
        hash.table[i].next=-1;
        hash.table[i].value=-1;
    }

    double t_start = 0.0;
    double t_end = 0.0;

    int nr_of_ins = HTABLE_FILL_RATIO*LV;

    int warmups = 0.1*nr_of_ins;
    double ins_time = 0.0;
    MPI_Request req[200];
    int64_t tmp_bigRandVal[3];
    int64_t recv_bigRandVal[3];

    srand(hash.r*1000000);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Status status;

    for(int m = 0; m < nr_of_ins; m++) {
        if(m == warmups) {
            MPI_Barrier(MPI_COMM_WORLD);
            t_start = MPI_Wtime();
        }

        tmp_bigRandVal[0] = bigRandVal();//elem
        tmp_bigRandVal[1] = hashfunc(tmp_bigRandVal[0]); // pos
        tmp_bigRandVal[2] = tmp_bigRandVal[1]/hash.tsize; // owner
        //sender send elem,insert_position and receiver rank ID, receiver does insert.
        // all processes send data simultaneously.
        // tag = receiver ID, send out hash.p message, but only one message will be received.
        for (int proc=0; proc<hash.p; proc++) {
            if (hash.r == proc) continue;
            MPI_Isend(&tmp_bigRandVal, 3, MPI_INT64_T, /*owner*/ proc, /*tag*/ proc,
                      MPI_COMM_WORLD, &req[proc]);
        }
        //receiver from any source, but tag needs to be == my rank id.
        for (int proc=1; proc<hash.p; proc++) {
            MPI_Recv(&recv_bigRandVal,3, MPI_INT64_T, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            //now other ranks are still waiting for m
            if (hash.r == recv_bigRandVal[2]) {
                insert(recv_bigRandVal,&hash);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_end = MPI_Wtime();
    ins_time = t_end - t_start;

    int size = count(&hash);
    assert(size == countSimple(&hash));

    if(hash.r == 0) {
        printf("%i\t%f\n",hash.p,ins_time);
    }


    MPI_Finalize();
}
