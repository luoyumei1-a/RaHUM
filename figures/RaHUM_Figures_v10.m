function RaHUM_Figures_v10(outDir, figureToDraw)
% Article-ordered MATLAB plotting source for RaHUM v10.
% Uses actual recorded TRAIN losses and strictly restored adapter diagnostics.
% Native replay diagnostics do not imply semantic correctness. Figure 13
% reproduces an author-supplied earlier-configuration confusion matrix verbatim;
% it is separate from the current PointLLM execution and K replay.
% Native checkpoint SHA256: f39fc35d0d766c86485f3c630448e63554f7dac5b75a969da1ee7a2b6b2bfa76
% Real feature-cache SHA256: 01f2e9c765fb4efef8f4df8f77e70adfb6bf4c73f4cf5683139d7615520720a0
% Exact source data and execution receipts accompany the code release.
% Usage: RaHUM_Figures_v10('figures','all')
% Actual K=8..128 strict replay SHA256: 7cf03a0138e9c4956b08d1aded176eb21b949d0d0d4b1b272c4318c4141e4d7d
% Article order: Fig2, Fig3, Fig10 (six scene panels), Fig11 (entropy),
% Fig12 (adapter time), Fig13 (classification matrix). The entropy and time
% figures are separate square figures, without subfigure. Legacy asset names
% remain stable for existing LaTeX links.
if nargin < 1 || isempty(outDir), outDir = fullfile(pwd,'figures'); end
if nargin < 2 || isempty(figureToDraw), figureToDraw = 'all'; end
if ~isfolder(outDir), mkdir(outDir); end
fontName = 'Times New Roman';
colors.radar = [211 72 54]/255;
colors.reference = [48 100 141]/255;
colors.light = [154 213 221]/255;
colors.grid = [.85 .87 .89];
colors.third = [95 178 205]/255;
frameUtility = [0.62843841314315796 0.63538521528244019 0.63587367534637451 0.6359182596206665 0.63593679666519165 0.63593536615371704 0.63593566417694092 0.63593566417694092; 0.59148234128952026 0.59340089559555054 0.59252470731735229 0.59215325117111206 0.59206527471542358 0.5920569896697998 0.5920565128326416 0.5920565128326416; 0.58132249116897583 0.58187490701675415 0.58142960071563721 0.58110767602920532 0.58102619647979736 0.58102089166641235 0.58102089166641235 0.58102089166641235];
entropyP10 = [1.9233527421951293 2.4570096731185913 2.583692502975464 2.6121158361434937 2.6190191507339478 2.6194457054138183 2.6194843292236327 2.6194846630096436];
entropyP90 = [2.0179047346115113 2.5726699590682984 2.7646418333053591 2.8399916172027586 2.8673958539962769 2.8695468425750734 2.8696116209030151 2.869611644744873];
% Full-precision observations. Populations are 42 geometrically supported
% groups from one three-frame recorded window, not 42 independent scenes.
utility = [0.657035231590271 0.60460424423217773 0.65224099159240723 0.62695789337158203 0.5911794900894165 0.61459189653396606 0.61647850275039673 0.66060584783554077 0.71605151891708374 0.60753971338272095 0.59205329418182373 0.61417770385742188 0.61618322134017944 0.64278149604797363 0.60123360157012939 0.50324314832687378 0.60993695259094238 0.59689199924468994 0.58757388591766357 0.55488240718841553 0.6355927586555481 0.55877900123596191 0.43875351548194885 0.61032354831695557 0.63731253147125244 0.63636744022369385 0.6143336296081543 0.64422935247421265 0.57884025573730469 0.63810467720031738 0.53983443975448608 0.57521259784698486 0.59119391441345215 0.57116425037384033 0.545482337474823 0.52939081192016602 0.57928425073623657 0.59589040279388428 0.61243796348571777 0.56753289699554443 0.63044387102127075 0.58370250463485718];
maxAttention = [0.19507764279842377 0.26739901304244995 0.19876320660114288 0.19930239021778107 0.24918480217456818 0.22649016976356506 0.22857329249382019 0.19498704373836517 0.21938714385032654 0.23033025860786438 0.25868478417396545 0.22008223831653595 0.23841746151447296 0.18220977485179901 0.19721364974975586 0.27843949198722839 0.19707103073596954 0.18221944570541382 0.22676578164100647 0.22731034457683563 0.20092985033988953 0.19469305872917175 0.20863601565361023 0.2198871523141861 0.2654607892036438 0.22156044840812683 0.22258728742599487 0.19361002743244171 0.19619525969028473 0.21639068424701691 0.21736578643321991 0.20596463978290558 0.25287041068077087 0.26966843008995056 0.20777355134487152 0.21668067574501038 0.24442248046398163 0.20658816397190094 0.18298013508319855 0.22958871722221375 0.2011561244726181 0.19530549645423889];
frameAttention = [0.19958017766475677 0.15944790840148926 0.15182366967201233 0.16457535326480865 0.093366928398609161 0.088767118752002716 0.084982916712760925 0.057455915957689285; 0.18348859250545502 0.19900885224342346 0.14593519270420074 0.13751634955406189 0.10065975785255432 0.096042878925800323 0.070241622626781464 0.067106775939464569; 0.18138228356838226 0.1819135993719101 0.15494643151760101 0.14451657235622406 0.10705883800983429 0.096928238868713379 0.06682993471622467 0.066424086689949036];
trainCE = [2.6411845684051514 2.4695336818695068; 2.6411845684051514 2.3584704399108887];
localK = [8 16 24 32 48 64 96 128];
entropy = [1.9824524465061368 2.4958344527653287 2.6687286297480264 2.7339890457334972 2.7593623740332469 2.7620890821729387 2.7623364244188582 2.76233815011524];
forwardMs = [42.087849000381539 64.396519999718294 49.319241999910446 47.734758001752198 63.421529001061572 72.444400000676978 109.68614800003706 135.96418700035429];
assert(numel(utility)==42 && numel(maxAttention)==42);
assert(max(abs(sum(frameAttention,2)-1)) < 1e-5);
assert(all(trainCE(:)>0));
order = {'fig02_claim_types','fig03_weather','fig10_memory_entropy', ...
    'fig11_adapter_time','fig12_reference_attention'};
if strcmpi(figureToDraw,'all')
    chosen = order;
elseif isnumeric(figureToDraw)
    chosen = order(figureToDraw);
else
    chosen = cellstr(string(figureToDraw));
end
scenePanelCount=0;
dataDir=fullfile(fileparts(mfilename('fullpath')),'recorded_scene_inputs');
for ii=1:numel(chosen)
    stem=chosen{ii};
    switch stem
        case 'fig02_claim_types'
            f = canvas(16.2,7.0,stem);
            ax=axes(f,'Position',[.105 .235 .36 .70]);
            histogram(ax,utility,.35:.05:.80,'FaceColor',colors.radar, ...
                'EdgeColor','w','FaceAlpha',1,'LineWidth',.8);
            xlabel(ax,'Radar utility'); ylabel(ax,'Groups');
            xlim(ax,[.35 .80]); xticks(ax,[.4 .5 .6 .7 .8]);
            ylim(ax,[0 22]); yticks(ax,[0 5 10 15 20]);
            styleAxes(ax,18);
            ax=axes(f,'Position',[.605 .235 .36 .70]);
            histogram(ax,maxAttention,.15:.025:.325,'FaceColor',colors.reference, ...
                'EdgeColor','w','FaceAlpha',1,'LineWidth',.8);
            xlabel(ax,'Maximum attention'); ylabel(ax,'Groups');
            xlim(ax,[.15 .325]); xticks(ax,[.15 .20 .25 .30]);
            ylim(ax,[0 22]); yticks(ax,[0 5 10 15 20]);
            styleAxes(ax,18);
        case 'fig03_weather'
            f=canvas(13.2,6.8,stem);
            ax=axes(f,'Position',[.145 .22 .81 .74]); hold(ax,'on');
            palette=[colors.radar;colors.reference;colors.third];
            marks={'o','s','^'};
            for frame=1:3
                plot(ax,1:numel(localK),frameUtility(frame,:),['-' marks{frame}], ...
                    'Color',palette(frame,:),'MarkerFaceColor',palette(frame,:), ...
                    'MarkerSize',6.5,'LineWidth',1.65);
            end
            xlabel(ax,'Local memory K'); ylabel(ax,'Mean radar utility');
            xlim(ax,[.7 numel(localK)+.3]); xticks(ax,1:numel(localK));
            xticklabels(ax,string(localK)); xtickangle(ax,0);
            ylim(ax,[.54 .66]); yticks(ax,.54:.03:.66);
            legend(ax,{'Frame 1','Frame 2','Frame 3'},'Location','southwest', ...
                'FontName',fontName,'FontSize',15,'Box','off','NumColumns',3);
            styleAxes(ax,18);
        case 'fig10_memory_entropy'
            f=canvas(12.0,11.4,stem);
            ax=axes(f,'Position',[.20 .18 .75 .77]); hold(ax,'on');
            plot(ax,1:numel(localK),entropyP90,'-^','Color',colors.third, ...
                'MarkerFaceColor',colors.third,'MarkerSize',7,'LineWidth',1.65);
            plot(ax,1:numel(localK),entropy,'-o','Color',colors.radar, ...
                'MarkerFaceColor',colors.radar,'MarkerSize',7,'LineWidth',1.65);
            plot(ax,1:numel(localK),entropyP10,'-s','Color',colors.reference, ...
                'MarkerFaceColor',colors.reference,'MarkerSize',7,'LineWidth',1.65);
            xticks(ax,1:numel(localK)); xticklabels(ax,string(localK));
            xtickangle(ax,0); xlim(ax,[.7 numel(localK)+.3]); ylim(ax,[0 3.5]);
            yticks(ax,0:1:3);
            xlabel(ax,'Local memory K'); ylabel(ax,'Attention entropy (nats)');
            legend(ax,{'P90','Mean','P10'},'Location','northwest', ...
                'FontName',fontName,'FontSize',17,'Box','off');
            styleAxes(ax,20);
        case 'fig11_adapter_time'
            f=canvas(12.0,11.4,stem);
            ax=axes(f,'Position',[.20 .18 .75 .77]);
            bar(ax,1:numel(localK),forwardMs,.67,'FaceColor',colors.reference, ...
                'EdgeColor','none');
            xticks(ax,1:numel(localK)); xticklabels(ax,string(localK));
            xtickangle(ax,0);
            xlim(ax,[.4 numel(localK)+.6]); ylim(ax,[0 150]); yticks(ax,0:50:150);
            xlabel(ax,'Local memory K'); ylabel(ax,'Adapter time (ms)');
            styleAxes(ax,20);
        case 'fig12_reference_attention'
            % Earlier author-reported matrix, reproduced verbatim from
            % the original author-provided plotting source, separate from native replay.
            f=canvas(18.8,9.5,stem);
            ax=axes(f); classes={'Car','Bike','Human','Van','Cone'};
            matrixData=[90.8 2.8 2.5 2.1 1.8; 3.4 87.6 4.7 2.6 1.7; ...
                2.9 6.2 85.8 2.8 2.3; 3.8 2.6 3.5 86.9 3.2; ...
                4.2 3.1 4.0 3.3 85.4];
            imagesc(ax,matrixData,[0 100]); axis(ax,'normal');
            ax.DataAspectRatioMode='auto'; ax.PlotBoxAspectRatioMode='auto';
            set(ax,'FontName',fontName,'FontSize',20,'FontWeight','normal');
            ax.XTick=1:5; ax.XTickLabel=classes;
            ax.YTick=1:5; ax.YTickLabel=classes;
            ax.TickLabelInterpreter='none'; xtickangle(ax,0);
            xlabel(ax,'Predicted','FontName',fontName,'FontSize',26,'FontWeight','normal');
            ylabel(ax,'True','FontName',fontName,'FontSize',26,'FontWeight','normal');
            for row=1:5
                for col=1:5
                    val=matrixData(row,col); textColor='k'; textWeight='normal';
                    if val>74, textColor='w'; end
                    if row==col, textWeight='bold'; end
                    text(ax,col,row,sprintf('%.1f',val),'HorizontalAlignment','center', ...
                        'VerticalAlignment','middle','FontWeight',textWeight, ...
                        'FontName',fontName,'FontSize',20,'Color',textColor);
                end
            end
            offDiagonal=matrixData; offDiagonal(1:size(matrixData,1)+1:end)=-inf;
            [~,idx]=max(offDiagonal(:)); [maxRow,maxCol]=ind2sub(size(offDiagonal),idx);
            rectangle(ax,'Position',[maxCol-.5 maxRow-.5 1 1], ...
                'EdgeColor',[.12 .12 .12],'LineStyle','--','LineWidth',1.8);
            colormap(ax,makeLightConfusionMap());
            cb=colorbar(ax); cb.Label.String='Row rate (%)';
            cb.Label.FontName=fontName; cb.Label.FontSize=22;
            cb.Label.FontWeight='normal'; cb.FontName=fontName;
            cb.FontSize=20; cb.FontWeight='normal'; cb.LineWidth=1.1;
            cb.Ticks=[0 25 50 75 100];
            set(ax,'LineWidth',1.65,'Box','on','YGrid','off','XGrid','off');
        otherwise
            error('Unknown figure: %s',stem);
    end
    set(findall(f,'-property','FontName'),'FontName',fontName);
    exportgraphics(f,fullfile(outDir,[stem '.pdf']),'ContentType','vector','BackgroundColor','white');
    savefig(f,fullfile(outDir,[stem '.fig']));
    fprintf('EXPORTED %s\n',stem);
    close(f);
    % Figure 10 follows Figure 3; preserve legacy native-chart asset names.
    if strcmpi(figureToDraw,'all') && strcmp(stem,'fig03_weather') && ...
            isfile(fullfile(dataDir,'CASE_SOURCE_MANIFEST.json'))
        scenePanelCount=renderRecordedInputs(dataDir,outDir);
    end
end
fid=fopen(fullfile(outDir,'MATLAB_execution.txt'),'w');
fprintf(fid,'MATLAB %s\nRelease %s\n',version,version('-release'));
fprintf(fid,'Native diagnostic populations: 42 supported radar groups, 3 frames, one feature window.\n');
fprintf(fid,'Additional recorded-input illustrations: %d panels from 3 separately identified scenes.\n',scenePanelCount);
fprintf(fid,'Checkpoint: 2 TRAIN updates; fixed weights across inference K.\n');
fprintf(fid,'Frame utility and entropy quantiles: strict K=8..128 replay, real cached features.\n');
fprintf(fid,'Local K timing: five repeats after warm-up, adapter-only CPU; 3296719 fixed parameters.\n');
fclose(fid);
    function cmap=makeLightConfusionMap()
        % Exact five anchors from the original author-provided confusion-plot code.
        anchors=[.98 .99 1; .83 .92 .97; .56 .76 .88; .95 .72 .58; .82 .32 .26];
        x=linspace(0,1,size(anchors,1)); xi=linspace(0,1,256);
        cmap=[interp1(x,anchors(:,1),xi).' interp1(x,anchors(:,2),xi).' ...
              interp1(x,anchors(:,3),xi).'];
    end
    function f=canvas(w,h,name)
        f=figure('Name',name,'Color','white','Units','centimeters', ...
            'Position',[2 2 w h],'PaperUnits','centimeters', ...
            'PaperPosition',[0 0 w h],'PaperSize',[w h],'Visible','off');
    end
    function styleAxes(ax,sz)
        set(ax,'FontName',fontName,'FontSize',sz,'LineWidth',1.25, ...
            'Box','on','TickDir','out','Layer','top','GridColor',colors.grid, ...
            'GridAlpha',.65,'YGrid','on','XGrid','off');
        ax.XLabel.FontSize=sz+2; ax.YLabel.FontSize=sz+2;
    end
end

function panelCount=renderRecordedInputs(dataDir,outDir)
% Render actual paired radar inputs and independent annotation references.
% These panels are NOT predictions from SOTA or RaHUM.
% Merge this function into the final article-ordered MATLAB source if the
% author chooses an input/annotation figure instead of a model-output figure.
if nargin<1, dataDir=pwd; end
if nargin<2, outDir=fullfile(dataDir,'input_panels'); end
if ~isfolder(outDir), mkdir(outDir); end
M=jsondecode(fileread(fullfile(dataDir,'CASE_SOURCE_MANIFEST.json')));
panelCount=2*numel(M.cases);
for i=1:numel(M.cases)
    c=M.cases(i); token=char(c.sample_token); stem=sprintf('scene%02d',i);
    radar=readmatrix(fullfile(dataDir,[token '_radar_ego_xyz.csv']));
    truth=readmatrix(fullfile(dataDir,[token '_gt_corners_ego.csv']));
    assert(size(radar,2)==3 && all(isfinite(radar),'all'));
    for mode=1:2
        f=figure('Color','white','Units','centimeters','Position',[2 2 11.2 6.8], ...
            'PaperUnits','centimeters','PaperPosition',[0 0 11.2 6.8], ...
            'PaperSize',[11.2 6.8],'Visible','off');
        ax=axes(f,'Position',[.14 .20 .82 .74]);hold(ax,'on');
        scatter(ax,-radar(:,2),radar(:,1),22,[48 100 141]/255,'filled', ...
            'MarkerEdgeColor','none');
        if mode==2
            ids=unique(truth(:,1));
            for j=1:numel(ids)
                b=truth(truth(:,1)==ids(j),:);
                assert(size(b,1)==8 && all(b(:,2)'==1:8));
                % Footprint of original source 3D annotation, not a detection.
                q=b([1 5 7 3 1],3:5);
                plot(ax,-q(:,2),q(:,1),'-','Color',[.45 .45 .45],'LineWidth',1.2);
            end
            suffix='_reference';
        else
            suffix='_radar';
        end
        xlim(ax,[-25 25]);ylim(ax,[0 65]);xticks(ax,[-20 0 20]);yticks(ax,[0 30 60]);
        xlabel(ax,'Lateral (m)');ylabel(ax,'Forward (m)');
        set(ax,'Color','white','FontName','Times New Roman','FontSize',18, ...
            'LineWidth',1.0,'Box','on','XGrid','off','YGrid','off','TickDir','out');
        ax.XLabel.FontSize=20;ax.YLabel.FontSize=20;
        exportgraphics(f,fullfile(outDir,[stem suffix '.pdf']), ...
            'ContentType','vector','BackgroundColor','white');
        savefig(f,fullfile(outDir,[stem suffix '.fig'])); fprintf('EXPORTED %s%s\n',stem,suffix); close(f);
    end
end
end
