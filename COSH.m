
function outlier_obj=COSH(data,label)
ini_time=cputime;
size(data);
k=length(unique(label));
[num_obj,num_att]=size(data);
num_value=0;
for i=1:num_att
    num_value_i=length(unique(data(:,i)));
    num_value=num_value+length(unique(data(:,i)));
end

%% build the value graph
g_value=zeros(num_value,num_value);
for i=1:num_obj
    for j=1:num_att
        for k=j:num_att
        index_j=data(i,j);
        index_k=data(i,k);
        if index_j==index_k
            g_value(index_j,index_k)=g_value(index_j,index_k)+1;
        else
            g_value(index_j,index_k)=g_value(index_j,index_k)+1;
            g_value(index_k,index_j)=g_value(index_k,index_j)+1;
        end
        end
    end
end

deg_table=diag(g_value); % which equals the |attribute_num|*frequency of each value
% G_ini = graph(g_value,'OmitSelfLoops');
% % G = graph(g_value,);
% h = plot(G_ini);
g_value;
%% feature mutual information
nmi_value=ones(num_value,num_value);
for i=1:num_att-1
    for j=i:num_att
        nmi_ij=NMI(data(:,i)',data(:,j)');
        nmi_value(unique(data(:,i)),unique(data(:,j)))=nmi_ij;
        nmi_value(unique(data(:,j)),unique(data(:,i)))=nmi_ij;
    end
end

%% build the value influence matrix
fre_table=deg_table;
occurrence_matrix=zeros(num_value,num_value);
frequency_matrix=zeros(num_value,num_value);
mi_sim_matrix=zeros(num_value,num_value);
for i =1:num_value
    for j=1:num_value
        if i==j
            occurrence_matrix(i,j)=1;
            frequency_matrix(i,j)=1;
            mi_sim_matrix(i,j)=1;
        else
            if g_value(i,j)==0
                occurrence_matrix(i,j)=0;
               % fre_sim_matrix(i,j)=min(fre_table(i),fre_table(j))/max(fre_table(i),fre_table(j));
                frequency_matrix(i,j)=fre_table(i)/fre_table(j);               
                mi_sim_matrix(i,j)=0;
            else
                occurrence_matrix(i,j)=g_value(i,j)/fre_table(i);
                mi_sim_matrix(i,j)=g_value(i,j)/(fre_table(i)*fre_table(j));  
               % fre_sim_matrix(i,j)=nmi_value(i,j)*min(fre_table(i),fre_table(j))/max(fre_table(i),fre_table(j));
                frequency_matrix(i,j)=nmi_value(i,j)*fre_table(i)/fre_table(j);
               
            end
        end
    end
end


%% build the value-cluster space based on k-means

outlier_matrix=[];
flag=1;
i=2;
while flag
%for i=2:cluster_times
    % conditional probability matrix
    [cluster_result,centroid]=kmeans(occurrence_matrix,i);
    outlier_vector=find_outlier_v2(occurrence_matrix',cluster_result,centroid);
    %outlier_vector=find_outlier(occurrence_matrix,cluster_result,centroid);
    outlier_matrix=horzcat(outlier_matrix,outlier_vector);
    i=i+1;
        
end

flag=1;
i=2;
while flag    
%for i=2:cluster_times
   % the frequency similarity matrix
    [cluster_result_fre,centroid]=kmeans(frequency_matrix,i);
    %%give outlier score for each value
    outlier_vector=find_outlier_v2(frequency_matrix,cluster_result_fre,centroid);
    %outlier_vector=find_outlier(frequency_matrix,cluster_result_fre,centroid);
    outlier_matrix=horzcat(outlier_matrix,outlier_vector);
    i=i+1;          
end

%% outlier score for each value
size(outlier_matrix);
%outlier_value=sum(outlier_matrix,2);
outlier_value=max(outlier_matrix,[],2);
size(outlier_value);
%% calculate the object embedding 
outlier_obj=zeros(num_obj,1);
for i=1:num_obj
    outlier_score=0;
    for j=1:num_att
        outlier_score=outlier_score+outlier_value(data(i,j));
%         if outlier_value(data(i,j))>outlier_score
%             outlier_score=outlier_value(data(i,j));
%         end        
        
    end
    outlier_obj(i,1)=outlier_score;
end




end




function outlier_vector=find_outlier(data,label,centroid)
%find outlier by distance between node and centroid
v_num=size(data,1);
dis_vec=zeros(v_num,2);
k=max(label);
dis_vec(:,2)=label;
for i=1:v_num
    c=dis_vec(i,2);
    dis_vec(i,1)=sqrt(sum((data(i,:) - centroid(c,:)) .^ 2));
end
for j=1:k
    sub_dis=dis_vec(label==j,1);
    if max(sub_dis)==0
        dis_vec(label==j,1)=1;
    else
        max_dis=max(sub_dis);
        dis_vec(label==j,1)=sub_dis./max_dis;
    end
    
end
outlier_vector=dis_vec(:,1);
t=0.9;
outlier_vector(outlier_vector<t)=0;
end
function outlier_vector=find_outlier_v1(data,label,centroid)
%find outlier by cluster size
v_num=size(data,1);
dis_vec=zeros(v_num,3);
k=max(label);
dis_vec(:,2)=label;
for i=1:v_num
    c=dis_vec(i,2);
    dis_vec(i,3)=sum((centroid(c,:)-data(i,:)));
end
ave_dis=zeros(k,1);
for i=1:k
    ave_dis(i,1)=abs(sum(dis_vec(label==i,3)))/length(label(label==i));
    dis_vec(label==i,1)=1/size(label(label==i),1);  
    %dis_vec(label==i,3)=dis_vec(label==i,3)-ave_dis(i,1);  
end
total_dis=sum(data,2);
max_dis=min(data)';  % used to be max(data)',modified on 2/5
dis_weight=max_dis./total_dis;
dis_vec(dis_vec(:,3)<0,3)=0;
%dis_vec(:,3)=dis_vec(:,3)-min(dis_vec(:,3));
outlier_vector=dis_vec(:,3).*dis_vec(:,1).*dis_weight
outlier_vector=dis_vec(:,3).*dis_vec(:,1);
% outlier_vector=dis_vec(:,1).*dis_weight;
% outlier_vector=dis_weight;
% outlier_vector=dis_vec(:,3);
%outlier_vector(outlier_vector<0)=0;
end


function cluster_matrix=vec2matrix(result)
row=length(result);
col=length(unique(result));
cluster_matrix=zeros(row,col);
uni=unique(result);
for i=1:col
    cluster_matrix(result==uni(i),i)=1;    
    %cluster_matrix(find(result==uni(i)),i)=1-length(find(result==uni(i)))/row; 
end
end
function [cluster_matrix,flag]=drop_cluster(cluster_matrix)
[~,col]=size(cluster_matrix);
index=[];
flag=1;
a=60;
for i=1:col
    label_c=unique(cluster_matrix(:,i));

    if length(find(cluster_matrix(:,i)==label_c(1)))<=1 || length(find(cluster_matrix(:,i)==label_c(2)))<=1
        index=[index,i];
    end
end
%cluster_matrix(:,index)=[];  Do not remove the cluster with only one value
if length(index)>ceil(col/a)
    flag=0;
end

end



function outlier_vector=find_outlier_v2(data,label,centroid)
%find outlier by cluster size
v_num=size(data,1);

dis_vec=zeros(v_num,3);
k=max(label);
value_c=zeros(v_num,k);
dis_vec(:,2)=label;
for i=1:v_num
    c=dis_vec(i,2);
    dis_vec(i,3)=sum((centroid(c,:)'-data(i,:)'));
    value_c(i,c)=max(sum((centroid(c,:)-data(i,:))),0);
end
ave_dis=zeros(k,1);
cluster_dis=zeros(k,k);
for i=1:k
    ave_dis(i,1)=abs(sum(dis_vec(label==i,3)))/length(label(label==i));
    dis_vec(label==i,1)=1/size(label(label==i),1);  
    for j=1:k
        %cluster_dis(i,j)=sqrt(sum((centroid(i,:)-centroid(j,:)).^2))/length(label(label==i));
        %cluster_dis(i,j)=max(0,sum((centroid(j,:)-centroid(i,:)))/length(label(label==i)));
        %cluster_dis(i,j)=sqrt(sum((centroid(i,:)-centroid(j,:)).^2));
        cluster_dis(i,j)=max(0,sum((centroid(j,:)-centroid(i,:))));
    end
    
end
total_dis=sum(data,2);
max_dis=min(data)';  % used to be max(data)',modified on 2/5
dis_weight=max_dis./total_dis;
dis_vec(dis_vec(:,3)<0,3)=0;
%dis_vec(:,3)=dis_vec(:,3)-min(dis_vec(:,3));
outlier_vector=dis_vec(:,3).*dis_vec(:,1).*dis_weight;
outlier_vector=dis_vec(:,3).*dis_vec(:,1);
outlier_vector=sum(value_c*cluster_dis,2);
end



















