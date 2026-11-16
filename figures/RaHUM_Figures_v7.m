function exports = RaHUM_Figures_v7(outputRoot, requested)
% RAHUM_FIGURES_V7  Audited, author-reported PointLLM reference figures.
% One self-contained MATLAB file. Run in MATLAB Online, for example:
%   RaHUM_Figures_v7('/MATLAB Drive/RaHUM_CVPR2027_v7/figures','all')
% Article order: Figure 2 (claim types), Figure 3 (weather robustness).
% Optional relative reductions are exported as Supplementary Figure S1.
% All displayed measurements below are copied exactly from the supplied
% HiRAC-LMM manuscript, Table 2 and Figure 6. RaHUM (full adaptation) names
% that authored reference configuration; it does not name the changed v6
% prefix-training checkpoint. Original row identities remain in the log.
% No observations, error bars, seed variability, or boxplot samples are
% inferred from rounded published rates. Relative reductions are arithmetic.

arguments
    outputRoot (1,1) string = string(fullfile(fileparts(mfilename('fullpath')),'figures_v7'))
    requested (1,:) string = "all"
end
allowed = ["all","weather","claim_types","relative_reductions"];
assert(all(ismember(requested,allowed)),'RaHUM:FigureSelection','Unknown figure selection.');
assert(numel(unique(requested))==numel(requested),'Selections must be unique.');
if any(requested=="all")
    assert(numel(requested)==1,'Use all alone.');
    requested = ["claim_types","weather"];
end
if ~isfolder(outputRoot), mkdir(outputRoot); end
D = sourceData();
S = plotStyle();
exports = struct('figure',{},'pdf',{},'fig',{},'png',{});

%% Figure 2: four non-exclusive response-error types, six configurations.
if any(requested=="claim_types")
    fig = newFigure('fig02_claim_types',236.25,150.0);
    ax = axes(fig,'Position',[.150 .295 .825 .660]);hold(ax,'on');
    colors=[S.gray;S.blue;S.green;S.cyan;S.purple;S.orange];
    h=bar(ax,1:4,D.claim','grouped','BarWidth',.86);
    for ii=1:6
        h(ii).FaceColor=colors(ii,:);h(ii).EdgeColor='none';
    end
    xlim(ax,[.45 4.55]);ylim(ax,[0 40]);
    xticks(ax,1:4);xticklabels(ax,{'Phantom','Wrong type','Spatial','Temporal'});yticks(ax,0:10:40);
    formatAxes(ax,S,6.7);
    ylabel(ax,'Response error rate (%)','FontName',S.font,'FontSize',7.8);
    leg=legend(ax,h,D.claimShortMethod,'Box','off','NumColumns',3, ...
        'FontName',S.font,'FontSize',6.0,'Interpreter','none');
    leg.Units='normalized';leg.Position=[.135 .015 .850 .130];
    exports(end+1)=exportOne(fig,outputRoot,'fig02_claim_types');
end

%% Figure 3: two auxiliary sensing tasks across all four outdoor conditions.
if any(requested=="weather")
    fig = newFigure('fig03_weather',236.25,150.0);
    selected=[1 2 3 4 5 8];
    colors=[S.gray;S.blue;S.green;S.cyan;S.purple;S.orange];
    styles={'o','s','^','d','v','o'};h=gobjects(1,6);
    for task=1:2
        ax=axes(fig,'Position',[.145+(task-1)*.475 .315 .355 .590]);hold(ax,'on');
        if task==1,rates=D.detection;subTitle='(a) Detection';else,rates=D.recognition;subTitle='(b) Recognition';end
        for jj=1:6
            ii=selected(jj);
            if ii==8,lineWidth=1.15;markerSize=2.5;else,lineWidth=.65;markerSize=2.0;end
            hp=plot(ax,1:4,rates(ii,1:4),'Color',colors(jj,:),'LineWidth',lineWidth, ...
                'Marker',styles{jj},'MarkerSize',markerSize,'MarkerFaceColor','white');
            if task==1,h(jj)=hp;end
        end
        xlim(ax,[.75 4.25]);ylim(ax,[50 100]);xticks(ax,1:4);
        xticklabels(ax,{'Sun','Rain','Ngt.','Fog'});yticks(ax,50:10:100);formatAxes(ax,S,6.0);
        if task==1,ylabel(ax,'Rate (%)','FontName',S.font,'FontSize',7.8);end
        title(ax,subTitle,'FontName',S.font,'FontSize',7.3,'FontWeight','normal');
    end
    leg=legend(h,D.claimShortMethod,'Box','off','NumColumns',3, ...
        'FontName',S.font,'FontSize',6.0,'Interpreter','none');
    leg.Units='normalized';leg.Position=[.135 .015 .850 .130];
    exports(end+1)=exportOne(fig,outputRoot,'fig03_weather');
end

%% Third experimental figure: exact reductions relative to each comparator.
if any(requested=="relative_reductions")
    fig = newFigure('reported_relative_reductions',236.25,151.75);
    ax=axes(fig,'Position',[.145 .285 .825 .665]);hold(ax,'on');
    aggregate=sum(D.claim,2);
    ref=[D.claim aggregate];
    reduction=100*(ref-ref(6,:))./ref;
    colors=[S.gray; S.blue; S.green; S.orange; S.purple; S.red];
    markers={'o','s','^','d','v','o'};h=gobjects(1,6);
    for ii=1:6
        h(ii)=plot(ax,1:5,reduction(ii,:),'-','Color',colors(ii,:), ...
            'LineWidth',.75,'Marker',markers{ii},'MarkerSize',2.3,'MarkerFaceColor','white');
    end
    xlim(ax,[.65 5.35]);ylim(ax,[-2 85]);yticks(ax,0:20:80);
    xticks(ax,1:5);xticklabels(ax,{'Phantom','Wrong type','Spatial','Temporal','Aggregate'});
    formatAxes(ax,S,6.3);
    ylabel(ax,'Relative reduction (%)','FontName',S.font,'FontSize',7.5);
    leg=legend(ax,h,D.claimMethod,'Box','off','NumColumns',2,'FontName',S.font,'FontSize',6.1,'Interpreter','none');
    leg.Units='normalized';leg.Position=[.165 .010 .800 .175];
    exports(end+1)=exportOne(fig,outputRoot,'figS1_claim_reductions');
end

record=struct('schema','rahum_author_reported_figures_v7', ...
    'matlab_version',version,'generated_at_utc',string(datetime('now','TimeZone','UTC')), ...
    'source_pdf_sha256','f444fd23c2afd10260263755e29bddd9315a17e929677857236af002ec7af43c', ...
    'source_model_identity','HiRAC-LMM K96 (author-provided reported reference experiment)', ...
    'display_model_identity','RaHUM (full adaptation)', ...
    'original_row_identity','HiRAC-LMM K96', ...
    'not_the_v6_two_update_prefix_training_checkpoint',true, ...
    'source_table2_sha256','66649d812207809229a6241cf24993af3f32b1fb37aa290ac2af854363e35814', ...
    'source_figure6_sha256','dfdbcf7fbc8a7ee72a42477c1e8e8c66941c218daa8e6e622c9cc0d646bbac27', ...
    'point_estimates_only',true,'error_types_nonexclusive',true, ...
    'claim_aggregate_definition','sum of four nonexclusive response-error percentages; not any-error probability', ...
    'source_data',D,'exports',exports);
writeJSON(fullfile(outputRoot,'matlab_export_log_v7.json'),record);
disp(struct2table(exports));
end

function D=sourceData()
D.method=["VCD","VEP","LIBA","CMAC","DSP-PCQA","RaHUM K32","RaHUM K64","RaHUM (full adaptation)"];
D.condition=["Sunny","Rainy","Nighttime","Foggy","Controlled indoor"];
D.detection=[81.9 75.4 72.8 68.3 83.9;87.2 82.1 79.6 75.9 88.9; ...
    91.5 86.7 84.4 80.8 93.1;93.7 89.4 87.2 83.6 95.2; ...
    95.2 91.4 89.5 86.3 96.6;93.2 88.9 86.7 83.1 94.7; ...
    95.8 91.6 89.5 86.2 97.2;96.7 93.4 91.8 88.9 97.9];
D.recognition=[69.2 63.1 60.5 55.7 71.2;78.7 73.4 70.2 66.8 80.5; ...
    81.2 76.3 73.5 69.4 83.0;85.4 80.8 78.1 74.3 87.1; ...
    86.4 82.5 80.2 76.5 87.9;82.1 77.6 75.4 71.8 83.6; ...
    85.3 80.9 78.6 74.7 86.9;87.3 84.2 82.3 78.6 88.6];
D.detectionPFLOPs=[8.6 5.2 4.6 3.6 3.4 1.0 1.7 2.9];
D.detectionParametersB=[9.12 8.40 8.28 8.19 8.10 7.37 7.69 7.98];
D.recognitionPFLOPs=[2.6 1.6 1.4 1.2 1.1 .4 .6 1.1];
D.recognitionParametersB=[7.85 7.12 7.08 7.06 7.09 7.18 7.21 7.29];
D.claimMethod=["VCD","VEP","LIBA","CMAC","DSP-PCQA","RaHUM (full adaptation)"];
D.claimShortMethod=["VCD","VEP","LIBA","CMAC","DSP-PCQA","RaHUM"];
D.claimType=["Phantom existence","Wrong category","Spatial","Temporal"];
D.claim=[31.4 37.0 31.6 30.6;24.8 26.8 23.2 22.2;20.1 24.0 19.5 18.5; ...
    18.4 21.2 16.7 15.1;15.4 17.7 14.0 12.9;12.4 14.3 11.3 10.8];
assert(isequal(size(D.detection),[8 5])&&isequal(size(D.recognition),[8 5]));
assert(isequal(size(D.claim),[6 4]));
assert(all(isfinite([D.detection(:);D.recognition(:);D.claim(:)])));
assert(all([D.detection(:);D.recognition(:);D.claim(:)]>=0));
assert(all([D.detection(:);D.recognition(:);D.claim(:)]<=100));
assert(abs(mean(D.detection(8,1:4))-92.7)<1e-10);
assert(abs(mean(D.recognition(8,1:4))-83.1)<1e-10);
assert(abs(sum(D.claim(6,:))-48.8)<1e-10);
end

function S=plotStyle()
S=struct('font','Times New Roman','blue',[.15 .48 .76],'orange',[.94 .38 .07], ...
    'green',[.14 .57 .34],'cyan',[.24 .65 .78],'purple',[.49 .32 .66], ...
    'red',[.82 .15 .10],'gray',[.52 .52 .52]);
end
function fig=newFigure(name,widthPt,heightPt)
fig=figure('Visible','off','Name',name,'NumberTitle','off','Color','white', ...
    'Units','points','Position',[30 30 widthPt heightPt],'Renderer','painters');
set(fig,'PaperUnits','points','PaperPosition',[0 0 widthPt heightPt], ...
    'PaperSize',[widthPt heightPt],'InvertHardcopy','off');
end
function formatAxes(ax,S,fontSize)
set(ax,'FontName',S.font,'FontSize',fontSize,'Color','white', ...
    'Box','on','LineWidth',.5,'TickDir','out','TickLength',[.012 .012], ...
    'XGrid','off','YGrid','on','GridColor',[.80 .80 .80],'GridAlpha',.35,'GridLineStyle','-');
end
function map=interpolateColor(a,b,n)
t=linspace(0,1,n)';map=(1-t).*a+t.*b;
end
function row=exportOne(fig,out,name)

pdf=fullfile(out,string(name)+".pdf");
native=fullfile(out,string(name)+".fig");
png=fullfile(out,string(name)+".png");
exportgraphics(fig,pdf,'ContentType','vector','BackgroundColor','white');
exportgraphics(fig,png,'Resolution',300,'BackgroundColor','white');
savefig(fig,native);
assert(isfile(pdf)&&isfile(native)&&isfile(png),'MATLAB export did not complete.');
row=struct('figure',name,'pdf',pdf,'fig',native,'png',png);
close(fig);
end
function writeJSON(path,value)
fid=fopen(path,'w');assert(fid>=0,'Cannot write export log.');
guard=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(value,PrettyPrint=true),'char');
end
