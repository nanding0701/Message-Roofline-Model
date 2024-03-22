#include "xmr.hpp"
#include "commons/mpi.hpp"

#define MAX_AVG_COLLISIONS 4
#define HTABLE_FILL_RATIO 0.001

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

void insert(t_hash* hash, int64_t elem) {
  int64_t pos = hashfunc(elem);
  int64_t owner = pos/hash->tsize;

  { // remote update
    // CAS if table position is available
    int64_t compare = -1;
    int64_t result = -2;
    int64_t offset /* in ints */ = (pos-owner*hash->tsize)*2;

    MPI_Compare_and_swap(&elem, &compare, &result, MPI_INT64_T, owner, /* target disp */ offset, hash->twin);
    MPI_Win_flush_local(owner, hash->twin);

    if(result != compare) { // we lost the CAS, no direct add

      int64_t offset = (pos-owner*hash->tsize);
      int64_t origin = 1; // add 1
      int64_t result = -1; // address of newly added record

      // grab remote location
      MPI_Fetch_and_op(/* origin */ &origin, /* result */ &result, MPI_INT64_T, owner, /* disp */ 0, MPI_SUM, hash->nwin);
      MPI_Win_flush_local(owner, hash->nwin);

      if(result >= hash->tsize+hash->size) {
        fprintf(stderr,"Failed to insert due to the OVERFLOW\n");
        return;
      }

      // put element in grabbed location
      MPI_Put(&elem, 1, MPI_INT64_T, owner, 2*(result), 1, MPI_INT64_T, hash->twin);

      // try to put directly as table link
      int64_t newOffset = result;

      int64_t oldOffset = -111;    // mine
      offset = (pos-owner*hash->tsize);

      // grab lastptr element and plug in ptr to new element
      MPI_Fetch_and_op(/* origin */ &newOffset, /* result */ &oldOffset, MPI_INT64_T, owner, /* disp */ offset, MPI_REPLACE, hash->lwin);
      MPI_Win_flush_local(owner, hash->lwin);

      compare = -1;
      result = -2;
      offset /* in ints */ = (pos-owner*hash->tsize)*2+1;

      // try to update table pointer directly (if it's the first update)
      MPI_Compare_and_swap(&newOffset, &compare, &result, MPI_INT64_T, owner, /* target disp */ offset, hash->twin);
      MPI_Win_flush_local(owner, hash->twin);

      if(result != compare) { // we lost the CAS for the direct pointer -- no direct access to table!

        MPI_Put(&newOffset, 1, MPI_INT64_T, owner, 2*(oldOffset)+1, 1, MPI_INT64_T, hash->twin);
        MPI_Win_flush(owner, hash->twin);
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

  MPI_Win_allocate((hash.tsize+hash.size)*sizeof(t_elem), sizeof(int64_t), MPI_INFO_NULL, MPI_COMM_WORLD, &hash.table, &hash.twin);
  MPI_Win_lock_all(0, hash.twin);

  MPI_Win_allocate(hash.tsize*sizeof(int64_t), sizeof(int64_t), MPI_INFO_NULL, MPI_COMM_WORLD, &hash.last, &hash.lwin);
  MPI_Win_lock_all(0, hash.lwin);

  hash.nextfree = hash.tsize; // next free element in heap
  MPI_Win_create(&hash.nextfree, sizeof(int64_t), 1, MPI_INFO_NULL, MPI_COMM_WORLD, &hash.nwin);
  MPI_Win_lock_all(0, hash.nwin);

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

  srand(hash.r*1000000);

  MPI_Barrier(MPI_COMM_WORLD);

  for(int m = 0; m < nr_of_ins; m++) {
    if(m == warmups) {
      MPI_Barrier(MPI_COMM_WORLD);
      t_start = MPI_Wtime();
    }
    insert(&hash, bigRandVal());
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t_end = MPI_Wtime();
  ins_time = t_end - t_start;

  int size = count(&hash);
  assert(size == countSimple(&hash));

  if(hash.r == 0) {
    printf("%i\t%f\n",hash.p,ins_time);
  }

  MPI_Win_unlock_all(hash.nwin);
  MPI_Win_free(&hash.nwin);

  MPI_Win_unlock_all(hash.lwin);
  MPI_Win_free(&hash.lwin);

  MPI_Win_unlock_all(hash.twin);
  MPI_Win_free(&hash.twin);

  MPI_Finalize();
}
