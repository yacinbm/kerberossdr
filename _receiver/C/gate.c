/* KerberosSDR Gate
 *
 * Copyright (C) 2018-2019  Carl Laufer, Tamás Pető
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <string.h>

#define CFN "_receiver/C/gate_control_fifo" // Name of the gate control fifo - Control FIFO name

int BUFFER_SIZE;

static sem_t trigger_sem, flush_sem;
static volatile int trigger=0, exit_flag=0, flush_flag=0;
pthread_t fifo_read_thread;   
void * fifo_read_tf(void* arg) 
{
/*          FIFO read thread function
 * 
 */
    fprintf(stderr,"FIFO read thread is running\n");    
    FILE * fd = fopen(CFN, "r"); // FIFO descriptor
    if(fd!=0)
        fprintf(stderr,"FIFO opened\n");
    else
        fprintf(stderr,"FIFO open error\n");
    uint8_t trigger_read;
    while(1){
        fread(&trigger_read, sizeof(trigger_read), 1, fd);
        if( (uint8_t) trigger_read == 1)
        {
            fprintf(stderr,"Trigger received\n");
            trigger++;
            sem_wait(&trigger_sem);          
        }            
        else if( (uint8_t) trigger_read == 2)
        {
            fprintf(stderr,"[ EXIT ] FIFO read thread exiting \n");
            exit_flag = 1;
            break;
        }
        else if ( (uint8_t) trigger_read == 3)
        {
            fprintf(stderr,"[ INFO ] Flush command receive, dump the current buffer...");
            flush_flag = 1;
            sem_wait(&flush_sem);
        }
    }
    fclose(fd);
    return NULL;
}


int main(int argc, char** argv)
{
    // Get buffer from args
    BUFFER_SIZE = atoi(argv[1]) * 1024 * 4;

    int read_size;
    uint8_t * buffer;    

    sem_init(&trigger_sem, 0, 0);  // Semaphore is unlocked
    sem_init(&flush_sem, 0, 0); 
    pthread_create(&fifo_read_thread, NULL, fifo_read_tf, NULL);
    
    // Allocate sample buffer
    buffer= malloc(BUFFER_SIZE*sizeof(uint8_t));

    fprintf(stderr,"Start gate control\n");    
    while(1)
    {        
        // Break if stdout of previous pipe is closed()
        if(feof(stdin))
            break;
        
        if(flush_flag == 1)
        {
            fflush(stdin);
            flush_flag = 0;
            sem_post(&flush_sem);
        }
            
        read_size = fread(buffer,sizeof(*buffer), BUFFER_SIZE, stdin);

        if(read_size>0)
        {
            if(trigger == 1)
            {

                fwrite(buffer, sizeof(*buffer), read_size, stdout);
                fflush(stdout);
                trigger --;
                sem_post(&trigger_sem);
            }            
            else
            {
                fprintf(stderr,"No trigger, dropping %d samples..\n", read_size);
            }
            
            if(exit_flag)
              break;
        }
        
    }    
    pthread_join(fifo_read_thread, NULL);
    sem_destroy(&trigger_sem);
    fprintf(stderr,"[ EXIT ] Gate control exited\n");
    return 0;
    
}

