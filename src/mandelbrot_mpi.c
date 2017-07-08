#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MASTER_RANK 0
#define MANDELBROT_TAG 0
#define END_WORK_CODE -1

/* Global Variables */
double c_x_min;
double c_x_max;
double c_y_min;
double c_y_max;
double pixel_width;
double pixel_height;
int iteration_max = 200;
int image_size;
unsigned char **image_buffer;
int i_x_max;
int i_y_max;
int max_chunk_size;
int image_buffer_size;

/* MPI variables */
int world_size;
int world_rank;


/* Data structure used to compute a chunk of mandelbrot points */
typedef struct {
    int size;
    int start_index;
} MANDELBROT_CHUNK;

void allocate_image_buffer ();
void init (int argc, char *argv[]);
void update_rgb_buffer (int iteration, int x, int y);
void write_to_file ();
void compute_mandelbrot ();
void receive_and_compute (MANDELBROT_CHUNK *chunks);
void distribute_work (MANDELBROT_CHUNK *chunks, int nchunks);
int pull_computed_chunk (MANDELBROT_CHUNK *chunks, 
        int *assigned_chunks);
int escape_iteration (double c_x, double c_y);
int *compute_mandelbrot_chunk (MANDELBROT_CHUNK ck);
MANDELBROT_CHUNK *create_chunks ();


int gradient_size = 16;
int colors[17][3] = {
    {66, 30, 15},
    {25, 7, 26},
    {9, 1, 47},
    {4, 4, 73},
    {0, 7, 100},
    {12, 44, 138},
    {24, 82, 177},
    {57, 125, 209},
    {134, 181, 229},
    {211, 236, 248},
    {241, 233, 191},
    {248, 201, 95},
    {255, 170, 0},
    {204, 128, 0},
    {153, 87, 0},
    {106, 52, 3},
    {0, 0, 0},
};


void allocate_image_buffer () {
    int rgb_size = 3;
    image_buffer = (unsigned char **) malloc (sizeof (unsigned char *)
            *image_buffer_size);
    for (int i = 0; i < image_buffer_size; i++) {
        image_buffer[i] = (unsigned char *) malloc (sizeof 
                (unsigned char) * rgb_size);
    }
}


void free_image_buffer () {
    int i;
    for (i = 0; i < image_buffer_size; i++)
        free (image_buffer[i]);
    free (image_buffer);
}

void init (int argc, char *argv[]) {
    if (argc < 6) {
        printf ("usage: ./mandelbrot_pth c_x_min c_x_max c_y_min \
                c_y_max image_size\n");
        printf ("examples with image_size = 11500:\n");
        printf ("    Full Picture:         ./mandelbrot_pth -2.5 1.5 \
                -2.0 2.0 11500\n");
        printf ("    Seahorse Valley:      ./mandelbrot_pth -0.8 -0.7 \
                0.05 0.15 11500\n");
        printf ("    Elephant Valley:      ./mandelbrot_pth 0.175 \
                0.375 -0.1 0.1 11500\n");
        printf ("    Triple Spiral Valley: ./mandelbrot_pth -0.188 \
                -0.012 0.554 0.754 11500\n");
        exit (0);
    }
    else {
        sscanf (argv[1], "%lf", &c_x_min);
        sscanf (argv[2], "%lf", &c_x_max);
        sscanf (argv[3], "%lf", &c_y_min);
        sscanf (argv[4], "%lf", &c_y_max);
        sscanf (argv[5], "%d", &image_size);
        i_x_max           = image_size;
        i_y_max           = image_size;
        image_buffer_size = image_size * image_size;
        pixel_width       = (c_x_max - c_x_min) / i_x_max;
        pixel_height      = (c_y_max - c_y_min) / i_y_max;
        /*if (argc > 6)*/
            /*sscanf (argv[6], "%d", &num_threads);*/
        /*if (argc > 7)*/
            /*sscanf (argv[7], "%d", &chunk_size);*/
    }
}


void update_rgb_buffer (int iteration, int x, int y) {
    int color;
    if (iteration == iteration_max) {
        color = 16;
        image_buffer[(i_y_max * y) + x][0] = color;
        image_buffer[(i_y_max * y) + x][1] = color;
        image_buffer[(i_y_max * y) + x][2] = color;
    }
    else {
        color = iteration % gradient_size;
        image_buffer[(i_y_max * y) + x][0] = colors[color][0];
        image_buffer[(i_y_max * y) + x][1] = colors[color][1];
        image_buffer[(i_y_max * y) + x][2] = colors[color][2];
    }
}


void write_to_file () {
    FILE *file;
    char *filename               = "output.ppm";
    char *comment                = "# ";
    int max_color_component_value = 255;
    file = fopen (filename, "wb");
    fprintf (file, "P6\n %s\n %d\n %d\n %d\n", comment,
            i_x_max, i_y_max, max_color_component_value);
    for (int i = 0; i < image_buffer_size; i++){
        fwrite (image_buffer[i], 1 , 3, file);
    };
    fclose (file);
};


void compute_mandelbrot () {
    int nchunks = i_y_max;
    MANDELBROT_CHUNK *chunks = create_chunks (nchunks);
    if (world_rank == MASTER_RANK)
        distribute_work (chunks, nchunks);
    else
        receive_and_compute (chunks);
    free (chunks);
}


void receive_and_compute (MANDELBROT_CHUNK *chunks) {
    int theres_work = 1;
    int *escape_iterations;
    while (theres_work) {
        MPI_Status status;
        int i, chunk_size;
        int tag = MANDELBROT_TAG;
        MPI_Recv (&i, 1, MPI_INT, MASTER_RANK, tag, MPI_COMM_WORLD,
                &status);

        if (i == END_WORK_CODE) {
            theres_work = 0;
            continue;
        }

        escape_iterations = compute_mandelbrot_chunk (chunks[i]);
        chunk_size = chunks[i].size;
        MPI_Send (&escape_iterations, chunk_size, MPI_UNSIGNED_CHAR,
                MASTER_RANK, tag, MPI_COMM_WORLD);
        free (escape_iterations);
    }
}


void distribute_work (MANDELBROT_CHUNK *chunks, int nchunks) {
    int worker;
    int tag = MANDELBROT_TAG;
    int sent_chunks = 0;
    int received_chunks = 0;
    int end_msg = END_WORK_CODE;
    int *assigned_chunks = malloc (world_size * sizeof (int));

    printf ("Distributing work\n");
    // First we should send chunks to every processor
    for (worker = 0; worker < world_size; worker++) {
        if (worker != MASTER_RANK) {
            MPI_Send (&sent_chunks, 1, MPI_INT, worker, tag, 
                    MPI_COMM_WORLD);
            assigned_chunks[worker] = sent_chunks++;
        }
        if (sent_chunks >= nchunks) break;
    }

    printf ("Done with first batch... now waiting to receive and send new jobs\n");
    // Now everytime any processor finish a chunk we should send
    // another
    while (received_chunks < nchunks) {
        worker = pull_computed_chunk (chunks, assigned_chunks);
        received_chunks++;
        if (sent_chunks < nchunks) {
            MPI_Send (&sent_chunks, 1, MPI_INT, worker, tag, 
                    MPI_COMM_WORLD);
            assigned_chunks[worker] = sent_chunks++;
        }
        else {
            MPI_Send (&end_msg, 1, MPI_INT, worker, tag, 
                    MPI_COMM_WORLD);
            sent_chunks++;
        }
    }

    // We should signal every processor that the job is done
    while (sent_chunks < nchunks + world_size) {
        MPI_Send (&end_msg, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
        sent_chunks++;
    }
}


int pull_computed_chunk (MANDELBROT_CHUNK *chunks, 
        int *assigned_chunks) {
    MANDELBROT_CHUNK ck;
    int src_rank, i_x, i_y, i;
    int tag = MANDELBROT_TAG;
    MPI_Status status;
    unsigned char *buf = malloc (world_size * sizeof (max_chunk_size));

    MPI_Recv (&buf, max_chunk_size, MPI_UNSIGNED_CHAR, 
            MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);

    src_rank = status.MPI_SOURCE;
    ck = chunks[assigned_chunks[src_rank]];
    printf ("Received chunk %d from procesor %d\n", src_rank, assigned_chunks[src_rank]);
    for (i = 0; i < ck.size; i++) {
        i_y = (i + ck.start_index) / i_x_max;
        i_x = (i + ck.start_index) % i_x_max;
        printf ("a\n");
        update_rgb_buffer (buf[i], i_x, i_y);
        printf ("b\n");
    }

    return src_rank;
}


int *compute_mandelbrot_chunk (MANDELBROT_CHUNK ck) {
    int i_y, i_x, i;
    int iteration;
    int chunk_start, chunk_end;
    double c_x, c_y;
    int *escape_iterations = malloc (ck.size * sizeof (int));
    chunk_start = ck.start_index;
    chunk_end = chunk_start + ck.size;
    for (i = chunk_start; i < chunk_end; i++) {
        int idx = i - chunk_start;
        i_y = i / i_x_max;
        i_x = i % i_x_max;
        c_x = c_x_min + i_x * pixel_width;
        c_y = c_y_min + i_y * pixel_height;
        if (fabs (c_y) < pixel_height / 2) {
            c_y = 0.0;
        };
        iteration = escape_iteration (c_x, c_y);
        escape_iterations[idx] = iteration;
    }
    return escape_iterations;
}


MANDELBROT_CHUNK *create_chunks (int nchunks) {
    int i, chunk_size, remainder_chunk_size;
    MANDELBROT_CHUNK *chunks;
    chunk_size = (i_y_max * i_x_max) / nchunks;
    remainder_chunk_size = (i_y_max * i_x_max) % nchunks;
    chunks = malloc (nchunks * sizeof (MANDELBROT_CHUNK));
    for (i = 0; i < nchunks; i++) {
        chunks[i].size = chunk_size;
        chunks[i].start_index = i * chunk_size;
    }
    chunks[nchunks - 1].size += remainder_chunk_size;
    max_chunk_size = chunks[nchunks - 1].size;
    return chunks;
}


int escape_iteration (double c_x, double c_y) {
    double z_x, z_y, z_x_squared, z_y_squared;
    double escape_radius_squared = 4;
    int iteration;
    z_x         = 0.0;
    z_y         = 0.0;
    z_x_squared = 0.0;
    z_y_squared = 0.0;
    for (iteration = 0; iteration < iteration_max && \
            ((z_x_squared + z_y_squared) < escape_radius_squared);
            iteration++) {
        z_y         = 2 * z_x * z_y + c_y;
        z_x         = z_x_squared - z_y_squared + c_x;
        z_x_squared = z_x * z_x;
        z_y_squared = z_y * z_y;
    }
    return iteration;
}


int main (int argc, char *argv[]) {
    init (argc, argv);
    MPI_Init (NULL, NULL);
    MPI_Comm_size (MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);

    if (world_rank ==  MASTER_RANK)
        allocate_image_buffer ();

    compute_mandelbrot ();

    if (world_rank == MASTER_RANK) {
        write_to_file ();
        free_image_buffer ();
    }
    return 0;
};
